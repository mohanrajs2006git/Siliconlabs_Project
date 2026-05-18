#include "tflite_micro_model_config.h"
#pragma once

#include "ml/accelerator/mvpv1/mvpv1_kernel_utils.hpp"
#include "ml/compiled_model/compiled_model.hpp"

// Tensor indices as defined by Tensorflow
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/pooling.cc#L31
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;


namespace npu_toolkit {
namespace mvpv1 {
namespace pooling2d
{
using namespace tflite;

constexpr const unsigned MVP_INPUT_ARRAY    = 0;
constexpr const unsigned MVP_OUTPUT_ARRAY   = 1;


bool max_pool_is_supported(
    TfLiteContext* context,
    const PoolParams& params,
    const RuntimeShape& input_shape,
    const RuntimeShape& output_shape
);

TfLiteStatus max_pool_calculate(
    TfLiteContext* context,
    const PoolParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    bool execute = true
);

void max_pool_calculate_ref(
    TfLiteContext* context,
    const ConvParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& output_shape, int8_t* output_data
);


bool average_pool_is_supported(
    TfLiteContext* context,
    const PoolParams& params,
    const RuntimeShape& input_shape,
    const RuntimeShape& output_shape
);

TfLiteStatus average_pool_calculate(
    TfLiteContext* context,
    const PoolParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    bool execute = true
);

void average_pool_calculate_ref(
    TfLiteContext* context,
    const ConvParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& output_shape, int8_t* output_data
);

TfLiteStatus calculate_compiled(
    TfLiteContext* context,
    const int8_t* input_data,
    int8_t* output_data
);


} // namespace pooling2d
} // namespace mvpv1
} // namespace npu_toolkit