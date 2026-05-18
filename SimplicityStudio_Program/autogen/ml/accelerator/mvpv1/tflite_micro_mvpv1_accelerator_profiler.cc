#include "tflite_micro_model_config.h"
#ifdef TFLITE_MICRO_PROFILER_ENABLED

#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_accelerator.hpp"
#include "ml/accelerator/mvpv1/mvpv1_program_recorder.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_debug.hpp"

#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
#include "mvp/driver.hpp"
#include "slsc_simulator.hpp"
#include "slsc_mvpv1_simulator.h"
#endif // TFLITE_MICRO_SIMULATOR_ENABLED




namespace npu_toolkit
{


static const char* PERFCNT_NAMES[] =
{
  "run",
  "cmd",
  "stall",
  "noop",
  "alu-active",
  "pipe-stall",
  "io-fence-stall",
  "load0-stall",
  "load1-stall",
  "store-stall",
  "bus-stall",
  "load0-ahb-stall",
  "load1-ahb-stall",
  "load0-fence-stall",
  "load1-fence-stall"
};


int TfliteMicroMvpv1Accelerator::get_profiler_loop_count()
{
  return 15/2;
}

void TfliteMicroMvpv1Accelerator::start_model_profiler(int loop_index)
{
  _current_loop_index = loop_index;
  program_counter = 0;

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  auto& driver = ::mvpv1::Driver::defaultInstance();
  using ProgramCallback = void (*)(const ::mvpv1::regif::ProgramContext*);
  driver.programCallback = reinterpret_cast<ProgramCallback>(mvp_program_callback);
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED
}

void TfliteMicroMvpv1Accelerator::stop_model_profiler(int loop_index)
{
}

void TfliteMicroMvpv1Accelerator::start_layer_profiler(int op_idx, profiling::Profiler* profiler)
{
  MVPV1_DEBUG_PRINTF("#########################################");
  MVPV1_DEBUG_PRINTF(TfliteMicroModelHelper::current_layer_name());

  _perfcnt_override_enabled[0] = false;
  _perfcnt_override_value[0] = 0;
  _perfcnt_override_enabled[1] = false;
  _perfcnt_override_value[1] = 0;
  _current_kernel_profiler = profiler;

  #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
  MVP->CFG_CLR = (_MVP_CFG_PERF1CNTSEL_MASK | _MVP_CFG_PERF0CNTSEL_MASK);
  MVP->CFG_SET = MVP_CFG_PERFCNTEN|((_current_loop_index*2) << _MVP_CFG_PERF0CNTSEL_SHIFT)|((_current_loop_index*2+1) << _MVP_CFG_PERF1CNTSEL_SHIFT);
  #else  // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
  MVP->CFG_CLR = _MVP_CFG_PERF0CNTSEL_MASK;
  MVP->CFG_SET = MVP_CFG_PERFCNTEN|_MVP_CFG_PERF0CNTSEL_RUN;
  #endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
}

void TfliteMicroMvpv1Accelerator::stop_layer_profiler(int op_idx, profiling::Profiler* profiler)
{
  if(profiler != nullptr)
  {
    #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    //Simulator::stop_op_profiler(profiler->name());
    #endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED

    if(MVP->EN & MVP_EN_EN)
    {
      #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
      const uint32_t percnt0 = _perfcnt_override_enabled[0] ? _perfcnt_override_value[0] : (uint32_t)MVP->PERF[0].CNT;
      const uint32_t percnt1 = _perfcnt_override_enabled[1] ? _perfcnt_override_value[1] : (uint32_t)MVP->PERF[1].CNT;
      if(_current_loop_index == 0)
      {
        profiler->stats().accelerator_cycles = percnt0;
        const int32_t mvp_prog_cycles = profiler->get_custom_stat("mvp_prog_cycles");
        profiler->parent()->increment_custom_stat("mvp_prog_cycles", mvp_prog_cycles);
        const int32_t dma_wait_cycles = profiler->get_custom_stat("dma_wait_cycles");
        profiler->parent()->increment_custom_stat("dma_wait_cycles", dma_wait_cycles);
      }
      profiler->parent()->increment_custom_stat(PERFCNT_NAMES[_current_loop_index*2], percnt0);
      profiler->parent()->increment_custom_stat(PERFCNT_NAMES[_current_loop_index*2+1], percnt1);
      profiler->increment_custom_stat(PERFCNT_NAMES[_current_loop_index*2], percnt0);
      profiler->increment_custom_stat(PERFCNT_NAMES[_current_loop_index*2+1], percnt1);
      #else
      profiler->stats().accelerator_cycles = _perfcnt_override_enabled[0] ? _perfcnt_override_value[0] : (uint32_t)MVP->PERF[0].CNT;
      #endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED

      MVP->EN_CLR = MVP_EN_EN;
    }
  }

  mvpv1_kernel_record_memory_stats();

  _current_kernel_profiler = nullptr;
}


void TfliteMicroMvpv1Accelerator::set_perfcnt_override(int index, uint32_t count)
{
  _perfcnt_override_enabled[index] = true;
  _perfcnt_override_value[index] = count;
}

void TfliteMicroMvpv1Accelerator::increment_perfcnt_override(int index, uint32_t count)
{
  _perfcnt_override_enabled[index] = true;
  _perfcnt_override_value[index] += count;
}


void TfliteMicroMvpv1Accelerator::increment_custom_profiling_stat(const char*name, int amount)
{
  if(_current_kernel_profiler != nullptr)
  {
    _current_kernel_profiler->increment_custom_stat(name, amount);
  }
}

void TfliteMicroMvpv1Accelerator::set_custom_profiling_stat(const char*name, int amount)
{
  if(_current_kernel_profiler != nullptr)
  {
    _current_kernel_profiler->set_custom_stat(name, amount);
  }
}

} // namespace npu_toolkit


#endif // TFLITE_MICRO_PROFILER_ENABLED