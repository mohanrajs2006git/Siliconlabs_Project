#include "tflite_micro_model_config.h"


#include "ml/accelerator/mvpv1/add.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_registration_helper.hpp"



namespace tflite {

extern int          cmsis_nn_add_get_init_size();
extern TfLiteStatus cmsis_nn_add_prepare(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_add_invoke(TfLiteContext* context, TfLiteNode* node);


namespace
{
using namespace npu_toolkit;




struct OpData
{
    KernelType kernel_type; // must come first
    tflite::ArithmeticParams op_params;

    float16_t input1_scaler;
    float16_t input2_scaler;
    float16_t output_scaler;
};


TfLiteStatus mvp_prepare(TfLiteContext* context, TfLiteNode* node)
{
    OpData& data = *static_cast<OpData*>(node->user_data);
    const auto params = static_cast<const TfLiteAddParams*>(node->builtin_data);
    auto& op_params = data.op_params;
   MicroContext* micro_context = GetMicroContext(context);
    TfLiteTensor* input1 =
        micro_context->AllocateTempInputTensor(node, kInputTensor1);
    NPU_TOOLKIT_ENSURE(context, input1 != nullptr);
    TfLiteTensor* input2 =
        micro_context->AllocateTempInputTensor(node, kInputTensor2);
    NPU_TOOLKIT_ENSURE(context, input2 != nullptr);
    TfLiteTensor* output =
        micro_context->AllocateTempOutputTensor(node, kOutputTensor);
    NPU_TOOLKIT_ENSURE(context, output != nullptr);

    auto input1_shape = GetTensorShape(input1);
    auto input2_shape = GetTensorShape(input2);
    auto output_shape = GetTensorShape(output);

    // 8bit -> 8bit general quantized path, with general rescalings
    op_params.input1_offset = -input1->params.zero_point;
    op_params.input2_offset = -input2->params.zero_point;
    op_params.output_offset = output->params.zero_point;
    op_params.left_shift = 20;

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
            context, params->activation, output,
            &op_params.quantized_activation_min,
            &op_params.quantized_activation_max));

    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(mvpv1_is_current_layer_compiled(context))
    {
        data.kernel_type = KernelType::Compiled;
    } else
    #endif // MVPV1_COMPILED_KERNELS_ENABLED
    #if defined(MVPV1_KERNELS_ENABLED)
    if(npu_toolkit::mvpv1::add::is_supported(
        context,
        op_params,
        input1_shape,
        input2_shape,
        output_shape
    ))
    {
        data.kernel_type = KernelType::Mvp;
        const float input1_scale = input1->params.scale;
        const float input2_scale = input2->params.scale;
        const float output_scale = output->params.scale;

        data.output_scaler = ::mvpv1::normalize_fp16(1 / output_scale);
        data.input1_scaler = ::mvpv1::normalize_fp16(input1_scale);
        data.input2_scaler = ::mvpv1::normalize_fp16(input2_scale);

        // This is used by the GSDK implementation
        *(float*)&op_params.input1_multiplier = input1_scale;
        *(float*)&op_params.input2_multiplier = input2_scale;
        *(float*)&op_params.output_multiplier = 1.0 / output_scale;
    } else
    #endif // MVPV1_KERNELS_ENABLED
    {
        // The current layer is not supported by the accelerator
        data.kernel_type = KernelType::None;
        return kTfLiteDelegateError;
    }


    return kTfLiteOk;
}


TfLiteStatus eval_int8(
    TfLiteContext* context, TfLiteNode* node,
    TfLiteAddParams* params,
    const OpData& data,
    const TfLiteEvalTensor* input1,
    const TfLiteEvalTensor* input2,
    TfLiteEvalTensor* output)
{
    auto& op_params      = data.op_params;
    auto input1_shape    = tflite::micro::GetTensorShape(input1);
    auto input1_data     = tflite::micro::GetTensorData<int8_t>(input1);
    auto input2_shape    = tflite::micro::GetTensorShape(input2);
    auto input2_data     = tflite::micro::GetTensorData<int8_t>(input2);
    auto output_shape    = tflite::micro::GetTensorShape(output);
    auto output_data     = tflite::micro::GetTensorData<int8_t>(output);

    #if defined(MVPV1_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Mvp)
    {
         return npu_toolkit::mvpv1::add::calculate(
            context,
            op_params,
            data.input1_scaler,
            data.input2_scaler,
            data.output_scaler,
            input1_shape, input1_data,
            input2_shape, input2_data,
            output_shape, output_data
        );
    } else
    #endif // MVPV1_KERNELS_ENABLED
    #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
    if(data.kernel_type == KernelType::Compiled)
    {
         return npu_toolkit::mvpv1::add::calculate_compiled(
            context,
            input1_data,
            input2_data,
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

    auto* params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);

    const auto input1 = tflite::micro::GetEvalInput(context, node, kInputTensor1);
    const auto input2 = tflite::micro::GetEvalInput(context, node, kInputTensor2);
    auto output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

    mvpv1_kernel_record_layer_tensor("input1", input1);
    mvpv1_kernel_record_layer_tensor("input2", input2);
    mvpv1_kernel_record_layer_tensor("output", output);

    switch (output->type)
    {
    case kTfLiteInt8:
        return eval_int8(context, node, params, data, input1, input2, output);

    default:
        TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.", TfLiteTypeGetName(output->type), output->type);
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
        #if MVPV1_KERNELS_ADD_FALLBACK_ENABLED == 1
        ,cmsis_nn_add_get_init_size()
        #endif
    );
}

TfLiteStatus prepare(TfLiteContext* context, TfLiteNode* node)
{
    return registration_prepare(
        context,
        node,
        mvp_prepare
        #if MVPV1_KERNELS_ADD_FALLBACK_ENABLED == 1
        ,cmsis_nn_add_prepare
        #endif
    );
}

TfLiteStatus invoke(TfLiteContext* context, TfLiteNode* node)
{
    return registration_invoke(
        context,
        node,
        mvp_invoke
        #if MVPV1_KERNELS_ADD_FALLBACK_ENABLED == 1
        ,cmsis_nn_add_invoke
        #endif
    );
}


}  // namespace


TFLMRegistration Register_ADD() {
    return tflite::micro::RegisterOp(
        init,
        prepare,
        invoke
    );
}

}  // namespace tflite
