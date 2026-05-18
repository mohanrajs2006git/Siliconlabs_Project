#include "tflite_micro_model_config.h"
#pragma once

#include "ml/third_party/tflm/tlcc-common.h"
#include "ml/third_party/tflm/tlc-builtin_op_data.h"
#include "ml/third_party/tflm/micro_allocator.h"

#include "ml/utils/profiling.hpp"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"

#ifndef TFLITE_MICRO_LOGGER_ENABLED
#error TFLITE_MICRO_LOGGER_ENABLED must be defined with TFLITE_MICRO_PROFILER_ENABLED
#endif


namespace npu_toolkit
{
class TfliteMicroModel;

class TfliteMicroProfiler
{
public:
  static TfliteMicroProfiler* create();
  void destroy();

  profiling::Profiler* root_profiler()
  {
    return _inference_profiler;
  }

  void register_callbacks(TfliteMicroModel* model);
  void unregister_callbacks(TfliteMicroModel* model);
  void set_loop_index(int index);

  bool calculate_op_metrics(
    TfLiteContext *context,
    const tflite::NodeAndRegistration& node_and_registration,
    profiling::Metrics& metrics
  );

  void reset();

private:
  int _n_layers = 0;
  int _loop_index = -1;
  profiling::Profiler *_inference_profiler = nullptr;
  profiling::Profiler **_layer_profilers = nullptr;

  TfliteMicroProfiler() = default;

  bool prepare_model();

  void register_profiler(
    int layer_index,
    tflite::BuiltinOperator opcode,
    const tflite::NodeAndRegistration& node_and_registration
  );

  void start_model();
  void stop_model();
  void start_layer(int index);
  void stop_layer(int index);

  static TfLiteStatus _section_callback(
    TfliteMicroSection section,
    bool is_start,
    void* arg
  );

  static TfLiteStatus _layer_callback(
    TfliteMicroSection section,
    bool is_start,
    int index,
    const tflite::NodeAndRegistration& node_and_registration,
    TfLiteStatus status,
    void* arg
  );
};


} // namespace npu_toolkit