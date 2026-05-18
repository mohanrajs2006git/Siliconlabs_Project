#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/fully_connected.hpp"


namespace npu_toolkit {
namespace mvpv1 {
namespace fully_connected
{

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    const int8_t* weights_data,
    const float16_t* bias_data,
    int8_t* output_data
)
{
  const float16_t zero_data(0);
  auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
  auto array_base_addresses = mvpv1_context.array_base_addresses;

  array_base_addresses[MVP_OUTPUT_ARRAY] = (uintptr_t)output_data;
  array_base_addresses[MVP_INPUT_ARRAY] = (uintptr_t)input_data;
  array_base_addresses[MVP_WEIGHTS_ARRAY] = (uintptr_t)weights_data;
  array_base_addresses[MVP_BIAS_ARRAY] = (uintptr_t)((bias_data == nullptr) ? &zero_data : bias_data);
  mvpv1_context.update_array_base_addresses_callback = nullptr;

  return compiled_model_process_layer_with_accelerator(context);
}


} // namespace fully_connected
} // namespace compiled
} // namespace npu_toolkit