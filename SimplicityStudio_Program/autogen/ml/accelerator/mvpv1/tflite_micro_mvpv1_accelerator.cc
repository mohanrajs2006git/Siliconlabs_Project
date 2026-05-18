#include "tflite_micro_model_config.h"
#include "em_device.h"
#include "ml/platform/ml_clock_helper.h"

#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_accelerator.hpp"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_context.hpp"
#include "ml/compiled_model/compiled_model_context.hpp"
#include "ml/accelerator/mvpv1/mvpv1_accelerator_utils.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/accelerator/mvpv1/mvpv1_program_recorder.hpp"

#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
#include "mvp/driver.hpp"
#endif
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recorder.hpp"
#endif


namespace npu_toolkit
{



const char* TfliteMicroMvpv1Accelerator::name() const
{
  return "mvpv1";
}

const char* TfliteMicroMvpv1Accelerator::description() const
{
  return "Matrix Vector Processor v1";
}


bool TfliteMicroMvpv1Accelerator::init()
{
  #ifdef __arm__
  ml_set_accelerator_clock_enabled(true);
  MVP->EN_SET = MVP_EN_EN;
  MVP->SWRST_SET = MVP_SWRST_SWRST;
  while(MVP->SWRST & MVP_SWRST_RESETTING)
  {
  }
  #endif // __arm__


  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  init_simulator();
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED

  #if !defined(MVPV1_COMPILED_KERNELS_ONLY)
  ::mvpv1::setLogger(TfliteMicroMvpv1Accelerator::mvp_driver_log_writer);
  #endif // MVPV1_COMPILED_KERNELS_ONLY

  return true;
}

void TfliteMicroMvpv1Accelerator::deinit(TfLiteContext *context)
{
  #if !defined(MVPV1_COMPILED_KERNELS_ONLY)
  ::mvpv1::setLogger(nullptr);
  #endif // MVPV1_COMPILED_KERNELS_ONLY
  auto compiled_context = (context != nullptr) ? CompiledModelContext::get<TfliteMicroMvpv1Context>(context) : nullptr;
  if(compiled_context != nullptr)
  {
    compiled_context->deinit();
  }

  mvpv1_kernel_record_reset();

  #ifndef TFLITE_MICRO_SIMULATOR_ENABLED
  MVP->EN_CLR = MVP_EN_EN;
  ml_set_accelerator_clock_enabled(false);
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED
}

TfliteMicroModelContext* TfliteMicroMvpv1Accelerator::create_context(
  TfLiteContext *context
)
{
  return TfliteMicroMvpv1Context::create(context);
}


void TfliteMicroMvpv1Accelerator::mvp_driver_log_writer(const char* fmt, va_list args)
{
  TfliteMicroModelStatus::instance().vissue(false, fmt, args);
}


void TfliteMicroMvpv1Accelerator::mvp_program_callback(const void *program)
{
  mvpv1_kernel_record_mvp_program_callback(program);

  #ifdef MVPV1_ACCELERATOR_DUMP_REGISTERS_ENABLED
  auto dump_registers_func = [](const void *program)
  {
    auto& accelerator = TfliteMicroMvpv1Accelerator::instance();
    MVPV1_DEBUG_PRINTF("------------------------------------");
    MVPV1_DEBUG_PRINTF("Program %d", accelerator.program_counter++);
    MVPV1_DEBUG_DUMP_PROGRAM(program);
  };

  dump_registers_func(program);
  #endif // MVPV1_ACCELERATOR_DUMP_REGISTERS_ENABLED
}



} // namespace npu_toolkit


#if defined(__arm__) && !defined(MVPV1_COMPILED_KERNELS_ONLY)
extern "C" void MVP_IRQHandler(void)
{
  mvpv1::default_MVP_IRQHandler();
}
#endif
