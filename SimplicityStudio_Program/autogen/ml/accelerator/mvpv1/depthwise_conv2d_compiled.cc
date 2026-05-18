#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/depthwise_conv2d.hpp"


namespace npu_toolkit {
namespace mvpv1 {
namespace depthwise_conv2d
{

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    const int8_t* filter_data,
    const float16_t* bias_data,
    float16_t* output_scaler,
    int8_t* output_data
)
{
  const float16_t zero_data(0);
  auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
  auto array_base_addresses = mvpv1_context.array_base_addresses;

  array_base_addresses[MVP_INPUT_ARRAY] = (uintptr_t)input_data;
  array_base_addresses[MVP_FILTER_ARRAY] = (uintptr_t)filter_data;
  array_base_addresses[MVP_BIAS_ARRAY] = (uintptr_t)((bias_data == nullptr) ? &zero_data : bias_data);
  array_base_addresses[MVP_OUTPUT_SCALER_ARRAY] = (uintptr_t)output_scaler;
  array_base_addresses[MVP_OUTPUT_ARRAY] = (uintptr_t)output_data;
  mvpv1_context.update_array_base_addresses_callback = nullptr;

  return compiled_model_process_layer_with_accelerator(context);
}


} // namespace depthwise_conv2d
} // namespace compiled
} // namespace npu_toolkit