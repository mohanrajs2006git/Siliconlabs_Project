#include "tflite_micro_model_config.h"
#pragma once

#include "ml/accelerator/mvpv1/mvpv1_kernel_utils.hpp"
#include "ml/compiled_model/compiled_model.hpp"


// Tensor indices as defined by Tensorflow
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/conv.cc#L33
constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;



namespace npu_toolkit {
namespace mvpv1 {
namespace conv2d
{

using namespace tflite;


constexpr const unsigned MVP_OUTPUT_ARRAY   = 0; // must be position 0, since fused relu reuses
constexpr const unsigned MVP_SCALED_INPUT_ARRAY = 0;

constexpr const unsigned MVP_INPUT_ARRAY    = 1;
constexpr const unsigned MVP_FILTER_ARRAY   = 2;
constexpr const unsigned MVP_BIAS_ARRAY     = 3;
constexpr const unsigned MVP_OUTPUT_SCALER_ARRAY = 4;




bool is_supported(
    TfLiteContext* context,
    const ConvParams& params,
    const RuntimeShape& input_shape,
    const RuntimeShape& filter_shape,
    const RuntimeShape& bias_shape,
    const RuntimeShape& output_shape
);


TfLiteStatus calculate(
    TfLiteContext* context,
    const ConvParams& params,
    const float16_t* output_scaler,
    float16_t* scaled_input_buffer,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float16_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    bool execute = true
);

void calculate_ref(
    TfLiteContext* context,
    const ConvParams& params, const float16_t* output_scaler,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float16_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data
);

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    float16_t* scaled_input_buffer,
    const int8_t* filter_data,
    const float16_t* bias_data,
    const float16_t* output_scaler,
    int8_t* output_data
);





} // namespace conv2d
} // namespace mvpv1
} // namespace npu_toolkit