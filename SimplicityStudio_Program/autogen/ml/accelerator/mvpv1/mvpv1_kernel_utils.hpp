#include "tflite_micro_model_config.h"
#pragma once

#include <cassert>
#include <cstdlib>
#include <string>

#include "ml/third_party/float16/float16.hpp"
#include "ml/third_party/tflm/tensor_ctypes.h"
#include "ml/third_party/tflm/tlki-common.h"
#include "ml/third_party/tflm/runtime_shape.h"
#include "ml/third_party/tflm/types.h"
#include "ml/third_party/tflm/common.h"
#include "ml/third_party/tflm/quantization_util.h"
#include "ml/third_party/tflm/kernel_util.h"
#include "ml/third_party/tflm/padding.h"
#include "ml/third_party/tflm/tlc-builtin_op_data.h"
#include "ml/third_party/tflm/tlmk-kernel_util.h"


#include "ml/tflite_micro_model/tflite_micro_utils.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_context.hpp"
#include "ml/tflite_micro_model/tflite_micro_accelerator.hpp"
#include "ml/accelerator/mvpv1/mvpv1_accelerator_utils.hpp"

#include "ml/accelerator/mvpv1/mvpv1_program_recorder.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_config.hpp"



#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
#include "mvp/driver.hpp"
#include "mvp/program/debug.hpp"
#include "mvp/builder_macros.hpp"
#include "mvp/regif/regif_types.hpp"
#endif

/**
 * This indicates that the NVP program register is constant
 * and does not change throughout program execution.
 *
 * This is used by the MVP model compiler when generating program "deltas".
*/
#define ConstRegister(regId) (::mvpv1::ALURegID)mvpv1_kernel_flag_alu_register_as_const((uint8_t)regId)

#undef SetRegister
/**
 * Override the SetRegister macro so that @ref npu_kernel_flag_alu_register_as_init()
 * is called. This is used by the MVP model compiler when generating program "deltas".
*/
#define SetRegister(regId, ...)  __program.setRegister(regId, __VA_ARGS__); mvpv1_kernel_flag_alu_register_as_init(regId);


#undef BeginProgramBatch
#define BeginProgramBatch(_driver, ...) \
{ \
    using namespace ::mvpv1; \
    uint8_t __defaultProgramBuffer[GetRequiredBufferSizeBytes(__VA_ARGS__)] = { 0 }; \
    ProgramBatchLinkedList __batch(execute ? &_driver : nullptr); \
    SetProgramBuffer(__defaultProgramBuffer, sizeof(__defaultProgramBuffer)); \
    if(execute) { \
       _driver.setCycleCounterEnabled(true); \
    }

#undef BeginProgram
#define BeginProgram() \
{ \
   uint8_t __defaultProgramBuffer2[GetRequiredBufferSizeBytes(1)] = { 0 }; \
   auto __programPtr = execute ? __batch.allocate(true) : (Program*)__defaultProgramBuffer2; { \
   auto& __program = *__programPtr; \
   ::mvpv1::ProgramBuilder __programBuilder(__programPtr);

#undef EndProgram
#define EndProgram()\
__programBuilder.endProgram(); \
if(__program.lastError != Error::None){return kTfLiteError;} \
if(execute) mvpv1_kernel_record_program_usage(__programBuilder); \
}

#undef RunProgram
#define RunProgram() \
if(execute) __batch.add(__programPtr); \
}

#undef EndProgramBatch
#define EndProgramBatch() \
if(execute) { \
  __batch.wait(); \
  mvpv1_set_perfcnt_override(0, DefaultDriver.getCycleCounter()); \
} }


#define ISSUE_KERNEL_NOT_SUPPORTED_WARNING() { \
  auto& status = TfliteMicroModelStatus::instance(); \
  if(!status.is_warning_detected()){ \
      status.issue(false, "Not supported"); \
  } \
}

#define NPU_TOOLKIT_ENSURE(context, x) \
if(!(x)){ \
  NPU_TOOLKIT_ISSUE_MODEL_WARNING("%s:%d %s was not true.", __FILE__, __LINE__, # x); \
  return kTfLiteError; \
}

#define NPU_TOOLKIT_ENSURE_EQ(context, a, b) \
if((a) != (b)) { \
  NPU_TOOLKIT_ISSUE_MODEL_WARNING("%s:%d %s != %s (%d != %d)", __FILE__, __LINE__, #a, #b, (a), (b)); \
  return kTfLiteError; \
}

#define NPU_TOOLKIT_ENSURE_LE(context, a, b) \
if((a) <= (b)) { \
  NPU_TOOLKIT_ISSUE_MODEL_WARNING("%s:%d %s > %s (%d > %d)", __FILE__, __LINE__, #a, #b, (a), (b)); \
  return kTfLiteError; \
}

#define NPU_TOOLKIT_ENSURE_NE(context, a, b) \
if((a) == (b)) { \
  NPU_TOOLKIT_ISSUE_MODEL_WARNING("%s:%d %s == %s (%d == %d)", __FILE__, __LINE__, #a, #b, (a), (b)); \
  return kTfLiteError; \
}



namespace npu_toolkit
{

constexpr const unsigned MAX_VECTOR_STRIDE = 2047;
constexpr const unsigned MAX_VECTOR_COUNT = 1024;
constexpr const unsigned MAX_ROW_STRIDE = 2047;
constexpr const unsigned MAX_ROW_LENGTH = 1024;
constexpr const unsigned MAX_COLUMN_STRIDE = 2047;
constexpr const unsigned MAX_COLUMN_LENGTH = 1024;

// Selected values give range up to int32 but not larger, so that final_scale_factor doesn't overflow
//  power-of-two chosen so that adds 0 error during accumulation
// Note: could generate a value based on filter_count to maximize range or even handle extremely large filter counts
constexpr const float ACCUMULATOR_MULTIPLIER = 1 << 15;
constexpr const float ACCUMULATOR_SCALER =  1.0f / ACCUMULATOR_MULTIPLIER;

constexpr const float FP16_MIN = -65504.0f;
constexpr const float FP16_MAX = 65504.0f;
constexpr const float FP16_SMALLEST = 0.000000059605f;


#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
using ProgramBatchLinkedList = ::mvpv1::ProgramBatchLinkedList;

/**
 * The size in bytes of an MVPv1-only "program"
*/
static_assert(MAX_VECTOR_STRIDE == ::mvpv1::MAX_VECTOR_STRIDE);
static_assert(MAX_VECTOR_COUNT == ::mvpv1::MAX_VECTOR_COUNT);
static_assert(MAX_ROW_STRIDE == ::mvpv1::MAX_ROW_STRIDE);
static_assert(MAX_ROW_LENGTH == ::mvpv1::MAX_ROW_LENGTH);
static_assert(MAX_COLUMN_STRIDE == ::mvpv1::MAX_COLUMN_STRIDE);
static_assert(MAX_COLUMN_LENGTH == ::mvpv1::MAX_COLUMN_LENGTH);
#endif

enum class KernelType : uint32_t
{
  None = 0,
  Reference,
  ReferenceBroadcast,
  Mvp,
  Compiled
};

template<typename T>
struct AutoBuffer
{
  T* ptr;

  AutoBuffer(size_t length)
  {
    ptr = static_cast<T*>(malloc(length*sizeof(T)));
    if(ptr == nullptr)
    {
      assert(!"Failed to allocate buffer");
    }
  }

  ~AutoBuffer()
  {
    if(ptr != nullptr)
    {
      free(ptr);
    }
  }
};


TfLiteStatus populate_convolution_quantization_params(
  TfLiteContext* context,
  const TfLiteTensor* input,
  const TfLiteTensor* filter,
  TfLiteTensor* output,
  float16_t* per_channel_scalers,
  int num_channels,
  float accumulator_multipler
);

// this is for the pad kernel
inline int OffsetNoCheck(const tflite::RuntimeShape& shape, int i0, int i1, int i2, int i3, int i4) {
  TFLITE_DCHECK_EQ(shape.DimensionsCount(), 5);
  const int* dims_data = reinterpret_cast<const int*>(shape.DimsData());
  return (((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3) * dims_data[4] + i4;
}

// Converts inference-style shape to legacy tflite::Dims<4>.
// Copied this function here after removed from tensorflow/tflite-micro repo
inline tflite::Dims<4> ToRuntimeDims(const tflite::RuntimeShape& array_shape) {
  tflite::Dims<4> result;
  const int dimensions_count = array_shape.DimensionsCount();
  TFLITE_CHECK_LE(dimensions_count, 4);
  int cum_prod = 1;
  for (int i = 0; i < 4; i++) {
    const int new_dim =
        (i < dimensions_count) ? array_shape.Dims(dimensions_count - 1 - i) : 1;
    result.sizes[i] = new_dim;
    result.strides[i] = cum_prod;
    cum_prod *= new_dim;
  }
  return result;
}

TfLiteStatus allocate_scaled_bias_tensor(
  TfLiteContext* context,
  const TfLiteTensor* bias,
  float16_t** scaled_tensor_ptr
);


template<typename T>
inline T clamp(T f, T min, T max)
{
    return std::min(std::max(f, min), max);
}

inline float16_t normalize_fp16(float f)
{
    return (float16_t)clamp(f, FP16_MIN, FP16_MAX);
}


struct SplitSize
{
  int factor = 0;
  int adjusted_size = 0;
  int remainder = 0;

  bool is_supported() const
  {
    return adjusted_size > 0;
  }
};

SplitSize check_and_split_size(int initial_size);
#define ENSURE_SIZE_IS_SUPPORTED(result) if(!result.is_supported()) return kTfLiteError

} // namespace npu_toolkit

