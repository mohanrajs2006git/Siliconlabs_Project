#include "tflite_micro_model_config.h"



#include "ml/accelerator/mvpv1/pooling2d.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_registration_helper.hpp"


namespace tflite {

extern int          cmsis_nn_pooling_get_init_size();
extern TfLiteStatus cmsis_nn_max_pool_prepare(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_max_pool_invoke(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_average_pool_prepare(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_average_pool_invoke(TfLiteContext* context, TfLiteNode* node);


namespace
{
using namespace npu_toolkit;


struct OpData
{
    KernelType kernel_type; // must come first
    int scratch_buffer_index;
    PoolParams op_params;
};



TfLiteStatus mvp_prepare(TfLiteContext* context, TfLiteNode* node, bool is_average_pool)
{
    OpData& data = *static_cast<OpData*>(node->user_data);
    const auto params = static_cast<const TfLitePoolParams*>(node->builtin_data);
    auto& op_params = data.op_params;
    MicroContext* micro_context = GetMicroContext(context);
    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kInputTensor);
    TfLiteTensor* output =
        micro_context->AllocateTempOutputTensor(node, kOutputTensor);
    const auto input_shape = GetTensorShape(input);
    const auto output_shape = GetTensorShape(output);

    // input: batch, height, width, channel
    int height = SizeOfDimension(input, 1);
    int width = SizeOfDimension(input, 2);

    int out_height, out_width;

    const auto padding = ComputePaddingHeightWidth(
            params->stride_height, params->stride_width,
            /*dilation_rate_height=*/1,
            /*dilation_rate_width=*/1, height, width, params->filter_height,
            params->filter_width, params->padding, &out_height, &out_width);

    op_params.stride_height = params->stride_height;
    op_params.stride_width = params->stride_width;
    op_params.filter_height = params->filter_height;
    op_params.filter_width = params->filter_width;
    op_params.padding_type = npu_toolkit::runtime_padding_type(params->padding);
    op_params.padding_values.height = padding.height;
    op_params.padding_values.width = padding.width;

    (void)CalculateActivationRangeQuantized(context, params->activation, output,
            &op_params.quantized_activation_min, &op_params.quantized_activation_max);


    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(mvpv1_is_current_layer_compiled(context))
    {
        data.kernel_type = KernelType::Compiled;
    } else
    #endif // MVPV1_COMPILED_KERNELS_ENABLED
    #if defined(MVPV1_KERNELS_ENABLED)
    if(is_average_pool &&
        npu_toolkit::mvpv1::pooling2d::average_pool_is_supported(
            context,
            op_params,
            input_shape,
            output_shape
    ))
    {
        data.kernel_type = KernelType::Mvp;
    }
    else if(!is_average_pool &&
        npu_toolkit::mvpv1::pooling2d::max_pool_is_supported(
            context,
            op_params,
            input_shape,
            output_shape
    ))
    {
        data.kernel_type = KernelType::Mvp;
    } else
    #endif // MVPV1_KERNELS_ENABLED
    {
        // The current layer is not supported by the accelerator
        data.kernel_type = KernelType::None;
        return kTfLiteDelegateError;
    }

    return kTfLiteOk;
}

TfLiteStatus mvp_max_pool_prepare(TfLiteContext* context, TfLiteNode* node)
{
    return mvp_prepare(context, node, false);
}

TfLiteStatus mvp_average_pool_prepare(TfLiteContext* context, TfLiteNode* node)
{
    return mvp_prepare(context, node, true);
}

TfLiteStatus eval_max_pool_int8(
    TfLiteContext* context, TfLiteNode* node,
    TfLitePoolParams* params,
    const OpData& data,
    const TfLiteEvalTensor* input,
    TfLiteEvalTensor* output
)
{
    auto& op_params = data.op_params;

    auto input_shape = tflite::micro::GetTensorShape(input);
    auto input_data  = tflite::micro::GetTensorData<int8_t>(input);
    auto output_shape = tflite::micro::GetTensorShape(output);
    auto output_data = tflite::micro::GetTensorData<int8_t>(output);

    #if defined(MVPV1_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Mvp)
    {
         return npu_toolkit::mvpv1::pooling2d::max_pool_calculate(
            context,
            op_params,
            input_shape, input_data,
            output_shape, output_data
        );
    } else
    #endif // MVPV1_KERNELS_ENABLED
    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Compiled)
    {
         return npu_toolkit::mvpv1::pooling2d::calculate_compiled(
            context,
            input_data,
            output_data
        );
    } else
    #endif // MVPV1_COMPILED_KERNELS_ENABLED

    return kTfLiteError;
}

TfLiteStatus mvp_max_pool_invoke(TfLiteContext* context, TfLiteNode* node)
{
    const OpData& data = *(static_cast<const OpData*>(node->user_data));

    // If this layer is not supported by the accelerator
    if(data.kernel_type == KernelType::None)
    {
        // Return a delegate error,
        // indicating that we should use the fallback implementation
        return kTfLiteDelegateError;
    }

    auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

    const TfLiteEvalTensor* input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
    TfLiteEvalTensor*       output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
    NPU_TOOLKIT_ENSURE(context, input  != nullptr);
    NPU_TOOLKIT_ENSURE(context, output != nullptr);


    switch (input->type)
    {
    case kTfLiteInt8:
        return eval_max_pool_int8(context, node, params, data, input, output);

    default:
        TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.", TfLiteTypeGetName(input->type), input->type);
        return kTfLiteError;
    }
}



TfLiteStatus eval_average_pool_int8(
    TfLiteContext* context, TfLiteNode* node,
    TfLitePoolParams* params,
    const OpData& data,
    const TfLiteEvalTensor* input,
    TfLiteEvalTensor* output
)
{
    auto& op_params = data.op_params;
    auto input_shape = tflite::micro::GetTensorShape(input);
    auto input_data  = tflite::micro::GetTensorData<int8_t>(input);
    auto output_shape = tflite::micro::GetTensorShape(output);
    auto output_data = tflite::micro::GetTensorData<int8_t>(output);

    #if defined(MVPV1_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Mvp)
    {
         return npu_toolkit::mvpv1::pooling2d::average_pool_calculate(
            context,
            op_params,
            input_shape, input_data,
            output_shape, output_data
        );
    } else
    #endif // MVPV1_KERNELS_ENABLED
    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Compiled)
    {
         return npu_toolkit::mvpv1::pooling2d::calculate_compiled(
            context,
            input_data,
            output_data
        );
    } else
    #endif // MVPV1_COMPILED_KERNELS_ENABLED

    return kTfLiteError;
}

TfLiteStatus mvp_average_pool_invoke(TfLiteContext* context, TfLiteNode* node)
{
    const OpData& data = *(static_cast<const OpData*>(node->user_data));

    // If this layer is not supported by the accelerator
    if(data.kernel_type == KernelType::None)
    {
        // Return a delegate error,
        // indicating that we should use the fallback implementation
        return kTfLiteDelegateError;
    }

    auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

    const TfLiteEvalTensor* input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
    TfLiteEvalTensor*       output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
    NPU_TOOLKIT_ENSURE(context, input  != nullptr);
    NPU_TOOLKIT_ENSURE(context, output != nullptr);


    switch (input->type)
    {
    case kTfLiteInt8:
        return eval_average_pool_int8(context, node, params, data, input, output);

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
        #if MVPV1_KERNELS_MAX_POOL2D_FALLBACK_ENABLED == 1 || MVPV1_KERNELS_AVERAGE_POOL2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_pooling_get_init_size()
        #endif
    );
}

TfLiteStatus average_pool_prepare(TfLiteContext* context, TfLiteNode* node)
{
    return registration_prepare(
        context,
        node,
        mvp_average_pool_prepare
        #if MVPV1_KERNELS_AVERAGE_POOL2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_average_pool_prepare
        #endif
    );
}

TfLiteStatus average_pool_invoke(TfLiteContext* context, TfLiteNode* node)
{
    return registration_invoke(
        context,
        node,
        mvp_average_pool_invoke
        #if MVPV1_KERNELS_AVERAGE_POOL2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_average_pool_invoke
        #endif
    );
}

TfLiteStatus max_pool_prepare(TfLiteContext* context, TfLiteNode* node)
{
    return registration_prepare(
        context,
        node,
        mvp_max_pool_prepare
        #if MVPV1_KERNELS_MAX_POOL2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_max_pool_prepare
        #endif
    );
}

TfLiteStatus max_pool_invoke(TfLiteContext* context, TfLiteNode* node)
{
    return registration_invoke(
        context,
        node,
        mvp_max_pool_invoke
        #if MVPV1_KERNELS_MAX_POOL2D_FALLBACK_ENABLED == 1
        ,cmsis_nn_max_pool_invoke
        #endif
    );
}


} // namespace


TFLMRegistration Register_AVERAGE_POOL_2D() {
    return tflite::micro::RegisterOp(
        init,
        average_pool_prepare,
        average_pool_invoke
    );
}

TFLMRegistration Register_MAX_POOL_2D() {
    return tflite::micro::RegisterOp(
        init,
        max_pool_prepare,
        max_pool_invoke
    );
}

} // namespace tflite