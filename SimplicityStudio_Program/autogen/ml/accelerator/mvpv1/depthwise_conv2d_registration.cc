#include "tflite_micro_model_config.h"

#include "ml/accelerator/mvpv1/depthwise_conv2d.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_registration_helper.hpp"


namespace tflite {

extern int          cmsis_nn_depthwise_conv_get_init_size();
extern TfLiteStatus cmsis_nn_depthwise_conv_prepare(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_depthwise_conv_invoke(TfLiteContext* context, TfLiteNode* node);


namespace
{
using namespace npu_toolkit;


struct OpData
{
    KernelType kernel_type; // must come first

    int scratch_buffer_index;
    tflite::DepthwiseParams op_params;

    union
    {
        int32_t* per_channel_output_multiplier;
        float16_t* per_channel_output_scaler;
    };

    union
    {
        float16_t* bias;
        int32_t* per_channel_output_shift;
    };
    int32_t input_zero_point;
    int32_t filter_zero_point;
    int32_t output_zero_point;
};


TfLiteStatus mvp_prepare(TfLiteContext* context, TfLiteNode* node)
{
    TFLITE_DCHECK(node->user_data != nullptr);
    TFLITE_DCHECK(node->builtin_data != nullptr);

    unsigned scratch_buffer_size = 0;
    OpData& data = *static_cast<OpData*>(node->user_data);
    const auto params = static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
    auto& op_params = data.op_params;

    MicroContext* micro_context = GetMicroContext(context);

    TfLiteTensor* output =
        micro_context->AllocateTempOutputTensor(node, kOutputTensor);
    TfLiteTensor* bias = (node->inputs->size == 3) ?
        micro_context->AllocateTempInputTensor(node, kBiasTensor) : nullptr;
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kInputTensor);
    TfLiteTensor* filter =
        micro_context->AllocateTempInputTensor(node, kFilterTensor);
    NPU_TOOLKIT_ENSURE(context, output != nullptr);
    NPU_TOOLKIT_ENSURE(context, input != nullptr);
    NPU_TOOLKIT_ENSURE(context, filter != nullptr);


    int width = SizeOfDimension(input, 2);
    int height = SizeOfDimension(input, 1);
    int filter_width = SizeOfDimension(filter, 2);
    int filter_height = SizeOfDimension(filter, 1);
    const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
    const auto input_shape = GetTensorShape(input);
    const auto filter_shape = GetTensorShape(filter);
    const auto bias_shape = GetTensorShape(bias);
    const auto output_shape = GetTensorShape(output);

    int unused_output_height, unused_output_width;
    const auto padding = ComputePaddingHeightWidth(
            params->stride_height, params->stride_width, 1, 1, height, width,
            filter_height, filter_width, params->padding, &unused_output_height,
            &unused_output_width);

    data.bias = nullptr;
    op_params.padding_type = npu_toolkit::runtime_padding_type(params->padding);
    op_params.padding_values.width = padding.width;
    op_params.padding_values.height = padding.height;
    op_params.stride_width = params->stride_width;
    op_params.stride_height = params->stride_height;
    op_params.dilation_width_factor = params->dilation_width_factor;
    op_params.dilation_height_factor = params->dilation_height_factor;
    op_params.depth_multiplier = params->depth_multiplier;
    op_params.input_offset = -input->params.zero_point;
    op_params.weights_offset = 0;
    op_params.output_offset = output->params.zero_point;

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context,
        params->activation,
        output,
        &op_params.quantized_activation_min,
        &op_params.quantized_activation_max
    ));

    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(mvpv1_is_current_layer_compiled(context))
    {
        data.kernel_type = KernelType::Compiled;
        // If possible, the bias is converted to float16
        // by the compiler. However, if that wasn't possible
        // then we need to manually convert it now
        if(bias != nullptr && bias->type == kTfLiteInt32)
        {
            TF_LITE_ENSURE_STATUS(allocate_scaled_bias_tensor(context, bias, &data.bias));
        }
    } else
    #endif // MVPV1_COMPILED_KERNELS_ENABLED
    #if defined(MVPV1_KERNELS_ENABLED)
    if(npu_toolkit::mvpv1::depthwise_conv2d::is_supported(
        context, op_params,
        input_shape,
        filter_shape,
        bias_shape,
        output_shape
    ))
    {
        data.kernel_type = KernelType::Mvp;
        TF_LITE_ENSURE_STATUS(allocate_scaled_bias_tensor(context, bias, &data.bias));
    } else
    #endif // MVPV1_KERNELS_ENABLED
    {
        // The current layer is not supported by the accelerator
        data.kernel_type = KernelType::None;
        return kTfLiteDelegateError;

    }


    data.per_channel_output_scaler = NPU_TOOLKIT_ALLOCATE_PLANNED_PERSISTENT_BUFFER(float16_t, num_channels);
    NPU_TOOLKIT_ENSURE(context, data.per_channel_output_scaler != nullptr);

    TF_LITE_ENSURE_STATUS(npu_toolkit::populate_convolution_quantization_params(
        context, input, filter, output,
        data.per_channel_output_scaler,
        num_channels, npu_toolkit::ACCUMULATOR_MULTIPLIER)
    );


    NPU_TOOLKIT_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);

    const auto* affine_quantization =
            static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    NPU_TOOLKIT_ENSURE(context, affine_quantization);
    NPU_TOOLKIT_ENSURE(context, affine_quantization->scale);
    NPU_TOOLKIT_ENSURE(context, affine_quantization->zero_point);

    NPU_TOOLKIT_ENSURE(context,
                    affine_quantization->scale->size == 1 ||
                    affine_quantization->scale->size ==
                            filter->dims->data[kDepthwiseConvQuantizedDimension]);
    NPU_TOOLKIT_ENSURE_EQ(context, affine_quantization->scale->size,
                        affine_quantization->zero_point->size);

    return kTfLiteOk;
}


TfLiteStatus eval_int8(
    TfLiteContext* context,
    TfLiteNode* node,
    TfLiteDepthwiseConvParams* params,
    const OpData& data,
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filters,
    const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output
)
{
    TfLiteStatus status = kTfLiteOk;
    auto& op_params     = (DepthwiseParams&)data.op_params;
    auto input_shape    = tflite::micro::GetTensorShape(input);
    auto input_data     =  tflite::micro::GetTensorData<int8_t>(input);
    auto filters_shape  = tflite::micro::GetTensorShape(filters);
    auto filters_data   =  tflite::micro::GetTensorData<int8_t>(filters);
    auto bias_shape     = tflite::micro::GetTensorShape(bias);
    auto output_shape   = tflite::micro::GetTensorShape(output);
    auto output_data    =  tflite::micro::GetTensorData<int8_t>(output);

    #if defined(MVPV1_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Mvp)
    {
         return npu_toolkit::mvpv1::depthwise_conv2d::calculate(
            context,
            op_params,
            data.per_channel_output_scaler,
            input_shape, input_data,
            filters_shape, filters_data,
            bias_shape, data.bias,
            output_shape, output_data
        );
    }
    #endif // MVPV1_KERNELS_ENABLED
    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Compiled)
    {
        const float16_t* bias_data =
            (bias != nullptr && bias->type == kTfLiteInt32) ? data.bias :
            tflite::micro::GetOptionalTensorData<float16_t>(bias);
         return npu_toolkit::mvpv1::depthwise_conv2d::calculate_compiled(
            context,
            input_data,
            filters_data,
            bias_data,
            data.per_channel_output_scaler,
            output_data
        );
    }
    #endif // MVPV1_COMPILED_KERNELS_ENABLED

    return kTfLiteError;
}

TfLiteStatus mvp_invoke(TfLiteContext* context, TfLiteNode* node)
{
    const OpData& data = *(static_cast<const OpData*>(node->user_data));

    // If this layer is not supported by the accelerator
    if(data.kernel_type == KernelType::None)
    {
        // Return a delegate error,
        // indicating that we should use the fallback implementation
        return kTfLiteDelegateError;
    }

    auto* params = reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);

    const auto input = tflite::micro::GetEvalInput(context, node, kInputTensor);
    const auto filter = tflite::micro::GetEvalInput(context, node, kFilterTensor);
    const auto bias   = NumInputs(node) == 3
                        ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
                        : nullptr;
    auto output       = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

    // Record the filter and bias tensors
    // NOTE: This must be done here where the memory plan is committed
    mvpv1_kernel_record_layer_tensor("filters", filter);
    mvpv1_kernel_record_layer_tensor("bias", bias);
    mvpv1_kernel_record_layer_tensor("input", input);
    mvpv1_kernel_record_layer_tensor("output", output);

    switch (input->type)
    {
    case kTfLiteInt8:
        return eval_int8(context, node, params, data, input, filter, bias, output);

    default:
        TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.", TfLiteTypeGetName(input->type), input->type);
        return kTfLiteError;
    }

}

void* init(TfLiteContext* context, const char* buffer, size_t length)
{
    return registration_init(
        context,
        buffer,
        length,
        sizeof(OpData)
        #if MVPV1_KERNELS_DEPTHWISE_CONV2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_depthwise_conv_get_init_size()
        #endif
    );
}

TfLiteStatus prepare(TfLiteContext* context, TfLiteNode* node)
{
    return registration_prepare(
        context,
        node,
        mvp_prepare
        #if MVPV1_KERNELS_DEPTHWISE_CONV2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_depthwise_conv_prepare
        #endif
    );
}

TfLiteStatus invoke(TfLiteContext* context, TfLiteNode* node)
{
    return registration_invoke(
        context,
        node,
        mvp_invoke
        #if MVPV1_KERNELS_DEPTHWISE_CONV2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_depthwise_conv_invoke
        #endif
    );
}

} // namespace



TFLMRegistration Register_DEPTHWISE_CONV_2D() {
    return tflite::micro::RegisterOp(
        init,
        prepare,
        invoke
    );
}

} // namespace tflite