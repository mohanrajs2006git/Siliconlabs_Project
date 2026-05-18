#include "tflite_micro_model_config.h"
#pragma once

#include <functional>
#include <vector>
#include <string>
#include "ml/third_party/tflm/micro_allocator.h"

#include "ml/utils/helpers.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"
#ifdef TFLITE_MICRO_PROFILER_ENABLED
#include "ml/utils/profiler.hpp"
#endif


namespace npu_toolkit
{


class TfliteMicroAccelerator
{
public:
  virtual const char* name() const = 0;
  virtual const char* description() const = 0;
  virtual bool init(){ return true; }
  virtual void deinit(TfLiteContext *context){}

  virtual TfliteMicroModelContext* create_context(
    TfLiteContext *context
  )
  {
    return TfliteMicroModelContext::create(context);
  }

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  virtual bool set_simulator_memory(const char* region, void* base_address, uint32_t length){ return true; };
  virtual bool invoke_simulator(const std::function<bool()>&func){ return func(); };
  virtual bool is_backend_enabled() const { return true; }
  virtual bool set_backend_enabled(bool enabled) { return false; }
  virtual bool is_simulator_enabled() const { return true; }
  virtual bool set_simulator_enabled(bool enabled) { return false; }
  virtual bool set_compilation_mode_enabled(bool enabled) { return false; }
  virtual bool is_compilation_mode_enabled() const { return false; }


  void set_layer_disable_filters(const std::vector<std::string>* filters);
  bool is_current_layer_disabled();
  static bool get_current_layer_disabled(TfLiteContext* context);
  #endif

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  virtual int  get_profiler_loop_count() { return 0; }
  virtual void start_model_profiler(int loop_count){}
  virtual void stop_model_profiler(int loop_count){};
  virtual void start_layer_profiler(int idx, profiling::Profiler* profiler){};
  virtual void stop_layer_profiler(int idx, profiling::Profiler* profiler){};
  #endif


private:

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  const std::vector<std::string>* _layer_disable_filters = nullptr;
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED
};


extern TfliteMicroAccelerator* register_tflite_micro_accelerator();
extern TfliteMicroAccelerator* get_tflite_micro_accelerator();
extern TfliteMicroAccelerator* DLL_EXPORT(get_registered_tflite_micro_accelerator());
extern TfliteMicroAccelerator* DLL_EXPORT(register_tflite_micro_accelerator(TfliteMicroAccelerator* accelerator));



} // namespace npu_toolkit