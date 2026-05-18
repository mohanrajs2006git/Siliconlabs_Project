#include "tflite_micro_model_config.h"
#pragma once


#include "ml/accelerator/mvpv1/mvpv1_kernel_utils.hpp"
#include "ml/compiled_model/compiled_model.hpp"


// Tensor indices as defined by Tensorflow
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/fully_connected.cc#L50
constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;



namespace npu_toolkit {
namespace mvpv1 {
namespace fully_connected
{

using namespace tflite;

constexpr const unsigned MVP_OUTPUT_ARRAY   = 0; // must be position 0, since fused relu reuses
constexpr const unsigned MVP_INPUT_ARRAY    = 1;
constexpr const unsigned MVP_WEIGHTS_ARRAY  = 2;
constexpr const unsigned MVP_BIAS_ARRAY     = 3;


bool is_supported(
    TfLiteContext* context,
    const RuntimeShape& input_shape,
    const RuntimeShape& weights_shape,
    const RuntimeShape& bias_shape,
    const RuntimeShape& output_shape
);


TfLiteStatus calculate(
    TfLiteContext* context,
    const FullyConnectedParams& params, const float16_t output_scaler,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& weights_shape, const int8_t* weights_data,
    const RuntimeShape& bias_shape, const float16_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    bool execute = true
);

void calculate_ref(
    TfLiteContext* context,
    const FullyConnectedParams& params, const float16_t output_scaler,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& weights_shape, const int8_t* weights_data,
    const RuntimeShape& bias_shape, const float16_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data
);

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    const int8_t* weights_data,
    const float16_t* bias_data,
    int8_t* output_data
);


} // namespace fully_connected
} // namespace mvpv1
} // namespace npu_toolkit