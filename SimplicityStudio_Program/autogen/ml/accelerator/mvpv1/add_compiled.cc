#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/add.hpp"


namespace npu_toolkit {
namespace mvpv1 {
namespace add
{

TfLiteStatus calculate_compiled(
    TfLiteContext *context,
    const int8_t* input1_data,
    const int8_t* input2_data,
    int8_t* output_data
)
{
  auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
  auto array_base_addresses = mvpv1_context.array_base_addresses;

  array_base_addresses[MVP_INPUT1_ARRAY] = (uintptr_t)input1_data;
  array_base_addresses[MVP_INPUT2_ARRAY] = (uintptr_t)input2_data;
  array_base_addresses[MVP_OUTPUT_ARRAY] = (uintptr_t)output_data;
  array_base_addresses[MVP_INPUT1_ARRAY_REMAINDER] = (uintptr_t)input1_data;
  array_base_addresses[MVP_INPUT2_ARRAY_REMAINDER] = (uintptr_t)input2_data;
  array_base_addresses[MVP_OUTPUT_ARRAY_REMAINDER] = (uintptr_t)output_data;
  mvpv1_context.update_array_base_addresses_callback = nullptr;


  return compiled_model_process_layer_with_accelerator(context);
}


} // namespace add
} // namespace compiled
} // namespace npu_toolkit