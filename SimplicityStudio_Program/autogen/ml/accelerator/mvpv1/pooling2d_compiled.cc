#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/pooling2d.hpp"


namespace npu_toolkit {
namespace mvpv1 {
namespace pooling2d
{

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    int8_t* output_data
)
{
  auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
  auto array_base_addresses = mvpv1_context.array_base_addresses;

  array_base_addresses[MVP_INPUT_ARRAY] = (uintptr_t)input_data;
  array_base_addresses[MVP_OUTPUT_ARRAY] = (uintptr_t)output_data;
  mvpv1_context.update_array_base_addresses_callback = nullptr;

  return compiled_model_process_layer_with_accelerator(context);
}


} // namespace pooling2d
} // namespace compiled
} // namespace npu_toolkit