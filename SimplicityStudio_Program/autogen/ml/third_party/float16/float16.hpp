#include "tflite_micro_model_config.h"
#pragma once


// Embedded must use build flag: -mfp16-format=ieee
#if (defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE))
typedef __fp16 float16_t;
#else

#include "half.hpp"

typedef half_float::half float16_t;

#endif