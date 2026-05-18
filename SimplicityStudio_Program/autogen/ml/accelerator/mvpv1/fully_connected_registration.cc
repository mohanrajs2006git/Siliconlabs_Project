#include "tflite_micro_model_config.h"


#include "ml/accelerator/mvpv1/fully_connected.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_registration_helper.hpp"



namespace tflite {

extern int          cmsis_nn_fully_connected_get_init_size();
extern TfLiteStatus cmsis_nn_fully_connected_prepare(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_fully_connected_invoke(TfLiteContext* context, TfLiteNode* node);


namespace
{
using namespace npu_toolkit;


struct OpData
{
    KernelType kernel_type; // must come first
    int scratch_buffer_index;

    int32_t output_activation_min;
    int32_t output_activation_max;
    int32_t input_zero_point;
    int32_t filter_zero_point;
    int32_t output_zero_point;
    float16_t output_scaler;
    float16_t *bias;
};

TfLiteStatus mvp_prepare(TfLiteContext* context, TfLiteNode* node)
{
    TFLITE_DCHECK(node->user_data != nullptr);
    TFLITE_DCHECK(node->builtin_data != nullptr);

    unsigned scratch_buffer_size = 0;
    OpData& data = *static_cast<OpData*>(node->user_data);
    const auto params = static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

    MicroContext* micro_context = GetMicroContext(context);

    TfLiteTensor* input =
            micro_context->AllocateTempInputTensor(node, kInputTensor);
    TfLiteTensor* weights = micro_context->AllocateTempInputTensor(
            node, kWeightsTensor);
    TfLiteTensor* bias = (node->inputs->size == 3) ?
        micro_context->AllocateTempInputTensor(node, kBiasTensor) : nullptr;
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
            node, kOutputTensor);
    NPU_TOOLKIT_ENSURE(context, output != nullptr);
    NPU_TOOLKIT_ENSURE(context, weights != nullptr);
    NPU_TOOLKIT_ENSURE(context, input != nullptr);


    NPU_TOOLKIT_ENSURE_EQ(context, input->type, output->type);
    TF_LITE_ENSURE_MSG(context, input->type == weights->type,
                       "Hybrid models are not supported on TFLite Micro.");

    const auto input_shape = GetTensorShape(input);
    const auto weights_shape = GetTensorShape(weights);
    const auto bias_shape = GetTensorShape(bias);
    const auto output_shape = GetTensorShape(output);
    data.bias = nullptr;
    data.input_zero_point = input->params.zero_point;
    data.filter_zero_point = weights->params.zero_point;
    data.output_zero_point = output->params.zero_point;

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context,
        params->activation,
        output,
        &data.output_activation_min,
        &data.output_activation_max
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
    if(npu_toolkit::mvpv1::fully_connected::is_supported(
        context,
        input_shape,
        weights_shape,
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

    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(
        GetQuantizedConvolutionMultipler(
            context,
            input,
            weights,
            bias,
            output,
            &real_multiplier
    ));
    data.output_scaler = normalize_fp16(real_multiplier * npu_toolkit::ACCUMULATOR_MULTIPLIER);

    return kTfLiteOk;
}


TfLiteStatus eval_int8(
    TfLiteContext* context,
    TfLiteNode* node,
    TfLiteFullyConnectedParams* params,
    const OpData& data,
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* weights,
    const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output
)
{
    tflite::FullyConnectedParams op_params;

    auto input_shape    = tflite::micro::GetTensorShape(input);
    auto input_data     = tflite::micro::GetTensorData<int8_t>(input);
    auto weights_shape  = tflite::micro::GetTensorShape(weights);
    auto weights_data   = tflite::micro::GetTensorData<int8_t>(weights);
    auto output_shape   = tflite::micro::GetTensorShape(output);
    auto output_data    = tflite::micro::GetTensorData<int8_t>(output);
    auto bias_shape     = tflite::micro::GetTensorShape(bias);

    op_params.input_offset = -data.input_zero_point;
    op_params.weights_offset = -data.filter_zero_point;
    op_params.output_offset = data.output_zero_point;
    op_params.quantized_activation_min = data.output_activation_min;
    op_params.quantized_activation_max = data.output_activation_max;

    #if defined(MVPV1_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Mvp)
    {
         return npu_toolkit::mvpv1::fully_connected::calculate(
            context,
            op_params,
            data.output_scaler,
            input_shape, input_data,
            weights_shape, weights_data,
            bias_shape, data.bias,
            output_shape, output_data
        );
    } else
    #endif // MVPV1_KERNELS_ENABLED
    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Compiled)
    {
        const float16_t* bias_data =
            (bias != nullptr && bias->type == kTfLiteInt32) ? data.bias :
            tflite::micro::GetOptionalTensorData<float16_t>(bias);
         return npu_toolkit::mvpv1::fully_connected::calculate_compiled(
            context,
            input_data,
            weights_data,
            bias_data,
            output_data
        );
    } else
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

    auto* params = reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

    const auto input = tflite::micro::GetEvalInput(context, node, kInputTensor);
    const auto weights = tflite::micro::GetEvalInput(context, node, kWeightsTensor);
    const auto bias   = NumInputs(node) == 3
                        ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
                        : nullptr;
    auto output       = tflite::micro::GetEvalOutput(context, node, kOutputTensor);


    // Record the filter and bias tensors
    // NOTE: This must be done here where the memory plan is committed
    mvpv1_kernel_record_layer_tensor("weights", weights);
    mvpv1_kernel_record_layer_tensor("bias", bias);
    mvpv1_kernel_record_layer_tensor("input", input);
    mvpv1_kernel_record_layer_tensor("output", output);

    switch (input->type)
    {
    case kTfLiteInt8:
        return eval_int8(context, node, params, data, input, weights, bias, output);

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
        #if MVPV1_KERNELS_FULLY_CONNECTED_FALLBACK_ENABLED == 1
        ,cmsis_nn_fully_connected_get_init_size()
        #endif
    );
}

TfLiteStatus prepare(TfLiteContext* context, TfLiteNode* node)
{
    return registration_prepare(
        context,
        node,
        mvp_prepare
        #if MVPV1_KERNELS_FULLY_CONNECTED_FALLBACK_ENABLED == 1
        ,cmsis_nn_fully_connected_prepare
        #endif
    );
}

TfLiteStatus invoke(TfLiteContext* context, TfLiteNode* node)
{
    return registration_invoke(
        context,
        node,
        mvp_invoke
        #if MVPV1_KERNELS_FULLY_CONNECTED_FALLBACK_ENABLED == 1
        ,cmsis_nn_fully_connected_invoke
        #endif
    );
}




} // namespace


TFLMRegistration Register_FULLY_CONNECTED() {
    return tflite::micro::RegisterOp(
        init,
        prepare,
        invoke
    );
}

} // namespace tflite