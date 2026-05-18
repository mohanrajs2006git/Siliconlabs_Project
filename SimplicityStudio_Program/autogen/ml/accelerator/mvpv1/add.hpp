#include "tflite_micro_model_config.h"
#pragma once


#include "ml/accelerator/mvpv1/mvpv1_kernel_utils.hpp"
#include "ml/compiled_model/compiled_model.hpp"


// Tensor indices as defined by Tensorflow
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/add.cc
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;


namespace npu_toolkit {
namespace mvpv1 {
namespace add
{

using namespace tflite;

constexpr const unsigned MVP_OUTPUT_ARRAY   = 0; // must be position 0, since fused relu reuses
constexpr const unsigned MVP_INPUT1_ARRAY   = 1;
constexpr const unsigned MVP_INPUT2_ARRAY   = 2;
constexpr const unsigned MVP_OUTPUT_ARRAY_REMAINDER = 3;
constexpr const unsigned MVP_INPUT1_ARRAY_REMAINDER = 4;
constexpr const unsigned MVP_INPUT2_ARRAY_REMAINDER = 5;



bool is_supported(
    TfLiteContext* context,
    const ArithmeticParams& params,
    const RuntimeShape& input1_shape,
    const RuntimeShape& input2_shape,
    const RuntimeShape& output_shape
);

TfLiteStatus calculate(
    TfLiteContext* context,
    const ArithmeticParams& params,
    const float16_t input1_scaler,
    const float16_t input2_scaler,
    const float16_t output_scaler,
    const RuntimeShape& input1_shape, const int8_t* input1_data,
    const RuntimeShape& input2_shape, const int8_t* input2_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    bool execute = true
);

void calculate_ref(
    TfLiteContext* context,
    const ArithmeticParams& params,
    const float16_t input1_scaler,
    const float16_t input2_scaler,
    const float16_t output_scaler,
    const RuntimeShape& input1_shape, const int8_t* input1_data,
    const RuntimeShape& input2_shape, const int8_t* input2_data,
    const RuntimeShape& output_shape, int8_t* output_data
);

TfLiteStatus calculate_compiled(
    TfLiteContext *context,
    const int8_t* input1_data,
    const int8_t* input2_data,
    int8_t* output_data
);


} // namespace add
} // namespace mvpv1
} // namespace npu_toolkit