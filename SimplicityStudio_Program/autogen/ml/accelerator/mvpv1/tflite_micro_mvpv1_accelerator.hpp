#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>
#include <cstdarg>

#include "ml/tflite_micro_model/tflite_micro_accelerator.hpp"


namespace npu_toolkit
{


class TfliteMicroMvpv1Accelerator : public TfliteMicroAccelerator
{
public:


  static TfliteMicroMvpv1Accelerator& instance()
  {
    return *reinterpret_cast<TfliteMicroMvpv1Accelerator*>(
      get_tflite_micro_accelerator()
    );
  }

  const char* name() const override;
  const char* description() const override;
  bool init() override;
  void deinit(TfLiteContext *context) override;
  TfliteMicroModelContext* create_context(
    TfLiteContext *context
  ) override;

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  void init_simulator();
  bool set_simulator_memory(const char* region, void* base_address, uint32_t length) override;
  bool invoke_simulator(const std::function<bool()>&func) override;
  bool is_backend_enabled() const override;
  bool set_backend_enabled(bool enabled) override;
  bool is_simulator_enabled() const override;
  bool set_simulator_enabled(bool enabled) override;
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  int  get_profiler_loop_count() override;
  void start_model_profiler(int loop_index) override;
  void stop_model_profiler(int loop_index) override;
  void start_layer_profiler(int op_idx, profiling::Profiler* profiler) override;
  void stop_layer_profiler(int op_idx, profiling::Profiler* profiler) override;
  void set_perfcnt_override(int index, uint32_t count);
  void increment_perfcnt_override(int index, uint32_t count);
  void increment_custom_profiling_stat(const char*name, int amount);
  void set_custom_profiling_stat(const char*name, int amount);
  #endif // TFLITE_MICRO_PROFILER_ENABLED

  int program_counter = 0;

protected:

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  int _current_loop_index = 0;
  uint32_t _perfcnt_override_value[2] = {0, 0};
  profiling::Profiler* _current_kernel_profiler = nullptr;
  bool _perfcnt_override_enabled[2] = {false, false};
  #endif // TFLITE_MICRO_PROFILER_ENABLED

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  #ifndef MVPV1_ACCELERATOR_BACKEND_ENABLED_DEFAULT_VALUE
  #define MVPV1_ACCELERATOR_BACKEND_ENABLED_DEFAULT_VALUE true
  #endif
  #ifndef MVPV1_ACCELERATOR_SIMULATOR_ENABLED_DEFAULT_VALUE
  #define MVPV1_ACCELERATOR_SIMULATOR_ENABLED_DEFAULT_VALUE true
  #endif
  bool _backend_enabled = MVPV1_ACCELERATOR_BACKEND_ENABLED_DEFAULT_VALUE;
  bool _simulator_enabled = MVPV1_ACCELERATOR_SIMULATOR_ENABLED_DEFAULT_VALUE;
  void set_exception_handler();
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED

  static void mvp_driver_log_writer(const char* fmt, va_list args);
  static void mvp_program_callback(const void *program);
};

} // namespace npu_toolkit