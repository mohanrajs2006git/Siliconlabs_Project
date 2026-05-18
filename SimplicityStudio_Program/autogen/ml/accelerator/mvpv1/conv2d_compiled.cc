#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/conv2d.hpp"


namespace npu_toolkit {

namespace mvpv1 {
namespace conv2d
{

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    float16_t* scaled_input_buffer,
    const int8_t* filter_data,
    const float16_t* bias_data,
    const float16_t* output_scaler,
    int8_t* output_data
)
{
  const float16_t zero_data(0);
  auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
  auto array_base_addresses = mvpv1_context.array_base_addresses;
  auto array_base_addresses2 = mvpv1_context.array_base_addresses2;


  array_base_addresses[MVP_SCALED_INPUT_ARRAY] = scaled_input_buffer != nullptr ? (uintptr_t)scaled_input_buffer : (uintptr_t)input_data;
  array_base_addresses[MVP_INPUT_ARRAY] = (uintptr_t)input_data;
  array_base_addresses[MVP_FILTER_ARRAY] = (uintptr_t)filter_data;
  array_base_addresses[MVP_BIAS_ARRAY] = (uintptr_t)((bias_data == nullptr) ? &zero_data : bias_data);
  array_base_addresses[MVP_OUTPUT_SCALER_ARRAY] = (uintptr_t)output_scaler;

  memcpy(array_base_addresses2, array_base_addresses, sizeof(mvpv1_context.array_base_addresses));

  array_base_addresses2[MVP_INPUT_ARRAY] = array_base_addresses[MVP_SCALED_INPUT_ARRAY];
  array_base_addresses2[MVP_OUTPUT_ARRAY] = (uintptr_t)output_data;

  mvpv1_context.update_array_base_addresses_callback = [](
    TfliteMicroMvpv1Context& ctx
  )
  {
    auto array_base_addresses = ctx.array_base_addresses;
    auto array_base_addresses2 = ctx.array_base_addresses2;
    memcpy(array_base_addresses, array_base_addresses2, sizeof(ctx.array_base_addresses));
  };

  return compiled_model_process_layer_with_accelerator(context);
}


} // namespace conv2d
} // namespace compiled
} // namespace npu_toolkit