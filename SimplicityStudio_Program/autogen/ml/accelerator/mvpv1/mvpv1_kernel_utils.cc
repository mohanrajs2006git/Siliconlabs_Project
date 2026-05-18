#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/mvpv1_kernel_utils.hpp"
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recorder.hpp"
#endif


 extern "C" void _assert(const char* message, const char* filename, unsigned line);

namespace npu_toolkit
{


TfLiteStatus populate_convolution_quantization_params(
  TfLiteContext* context,
  const TfLiteTensor* input,
  const TfLiteTensor* filter,
  TfLiteTensor* output,
  float16_t* per_channel_scalers,
  int num_channels,
  float accumulator_multipler
)
{
    auto affine_quantization =
        reinterpret_cast<const TfLiteAffineQuantization*>(filter->quantization.params);

  // Populate multiplier and shift using affine quantization.
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;

  for (int i = 0; i < num_channels; ++i)
  {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float filter_scale = filter_scales[i];
    const float effective_output_scale = (input_scale * filter_scale) / output_scale;
    const float acc_output_scale = effective_output_scale * accumulator_multipler;
    per_channel_scalers[i] = normalize_fp16(acc_output_scale);
  }

  return kTfLiteOk;
}

TfLiteStatus allocate_scaled_bias_tensor(
  TfLiteContext* context,
  const TfLiteTensor* bias,
  float16_t** scaled_tensor_ptr
)
{
  if(bias != nullptr && bias->type == kTfLiteInt32)
  {
    const int bias_count = bias->bytes / sizeof(int32_t);
    auto scaled_bias_tensor = NPU_TOOLKIT_ALLOCATE_PERSISTENT_BUFFER(float16_t, bias_count);
    if(scaled_bias_tensor == nullptr)
    {
      return kTfLiteError;
    }

    auto src = tflite::GetTensorData<int32_t>(bias);
    auto dst = scaled_bias_tensor;
    for(int i = bias_count; i > 0; --i)
    {
        const auto b = *src++;
        *dst++ = float16_t(b * npu_toolkit::ACCUMULATOR_SCALER);
    }

    #ifdef TFLITE_MICRO_RECORDER_ENABLED
    TfliteMicroRecorder::record_data(
      "scaled_bias",
      (const void*)scaled_bias_tensor,
      sizeof(float16_t)*bias_count
    );
    #endif

    *scaled_tensor_ptr = scaled_bias_tensor;
  }
  else
  {
    *scaled_tensor_ptr = nullptr;
  }

  return kTfLiteOk;
}


SplitSize check_and_split_size(int initial_size)
{
  SplitSize result;
  int adjusted_size = initial_size;
  int factor = 1;
  int remainder = 0;

  while(adjusted_size > (int)MAX_COLUMN_LENGTH) {
    adjusted_size /= 2;
    factor *= 2;
  }
 
  if (factor * adjusted_size != initial_size) {
    remainder = initial_size - (factor * adjusted_size);
  }
 
  result.factor = factor;
  result.adjusted_size = adjusted_size;
  result.remainder = remainder;

  return result;
}


} // namespace npu_toolkit