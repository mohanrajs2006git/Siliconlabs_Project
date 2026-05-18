#include "tflite_micro_model_config.h"
#pragma once
#include <cstdint>

#include "ml/third_party/tflm/common.h"
#include "ml/accelerator/mvpv1/mvpv1_kernel_debug.hpp"



namespace npu_toolkit
{

/**
 * Return if the model has been compiled
 */
bool mvpv1_is_model_compiled(TfLiteContext *context);

/**
 * Return if the current ML model layer has pre-compiled data in the .tflite model file
*/
bool mvpv1_is_current_layer_compiled(TfLiteContext *context);


#ifdef TFLITE_MICRO_PROFILER_ENABLED
/**
 * Set the accelerator cycles to be a specific count.
*/
void mvpv1_set_perfcnt_override(int index, uint32_t count);
void mvpv1_increment_perfcnt_override(int index, uint32_t count);

/**
 * Increment/decrement a custom profiling stat by key
 * This is mostly used by the Python profiling scripts
*/
void mvpv1_increment_custom_profiling_stat(const char*name, int amount=1);
void mvpv1_set_custom_profiling_stat(const char*name, int amount);
#else

#define mvpv1_set_perfcnt_override(...)
#define mvpv1_increment_perfcnt_override(...)
#define mvpv1_increment_custom_profiling_stat(...)
#define mvpv1_set_custom_profiling_stat(...)


#endif // TFLITE_MICRO_PROFILER_ENABLED


} // namespace npu_toolkit
