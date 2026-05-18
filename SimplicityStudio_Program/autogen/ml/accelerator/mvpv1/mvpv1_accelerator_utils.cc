#include "tflite_micro_model_config.h"

#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_accelerator.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_context.hpp"
#include "ml/compiled_model/compiled_model_context.hpp"
#include "ml/accelerator/mvpv1/mvpv1_accelerator_utils.hpp"
#include "ml/accelerator/mvpv1/mvpv1_program_recorder.hpp"


namespace npu_toolkit
{


#ifdef TFLITE_MICRO_PROFILER_ENABLED
void mvpv1_set_perfcnt_override(int index, uint32_t count)
{
  TfliteMicroMvpv1Accelerator::instance().set_perfcnt_override(index, count);
}

void mvpv1_increment_perfcnt_override(int index, uint32_t count)
{
  if(index == 0)
  {
    mvpv1_kernel_record_program_run_cycles(count);
  }
  TfliteMicroMvpv1Accelerator::instance().increment_perfcnt_override(index, count);
}

void mvpv1_increment_custom_profiling_stat(const char*name, int amount)
{
  TfliteMicroMvpv1Accelerator::instance().increment_custom_profiling_stat(name, amount);
}

void mvpv1_set_custom_profiling_stat(const char*name, int amount)
{
  TfliteMicroMvpv1Accelerator::instance().set_custom_profiling_stat(name, amount);
}
#endif // TFLITE_MICRO_PROFILER_ENABLED


bool mvpv1_is_model_compiled(TfLiteContext *context)
{
  return CompiledModelContext::get<TfliteMicroMvpv1Context>(context) != nullptr;
}

bool mvpv1_is_current_layer_compiled(TfLiteContext *context)
{
    auto compiled_context = CompiledModelContext::get<TfliteMicroMvpv1Context>(context);
    if(compiled_context == nullptr)
    {
      return false;
    }

    const auto accelerator_id = compiled_context->get_layer_accelerator(
        TfliteMicroModelHelper::current_layer_index(context)
    );

    return accelerator_id != CompiledAcceleratorId::NoAccelerator;
}






} // namespace npu_toolkit