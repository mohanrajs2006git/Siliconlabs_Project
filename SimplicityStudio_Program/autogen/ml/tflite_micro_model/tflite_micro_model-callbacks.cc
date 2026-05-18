#include "tflite_micro_model_config.h"


#include "ml/utils/string.hpp"
#include "tflite_micro_model.hpp"
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recorder.hpp"
#endif
#ifdef TFLITE_MICRO_PROFILER_ENABLED
#include "ml/platform/ml_clock_helper.h"
#endif


namespace npu_toolkit
{


bool TfliteMicroModel::add_section_callback(
  TfliteMicroSectionCallback callback,
  void *arg
)
{
  return _section_callbacks.push_back({callback, arg}, true);
}

bool TfliteMicroModel::remove_section_callback(
  TfliteMicroSectionCallback callback,
  void *arg
)
{
  return _section_callbacks.remove({callback, arg});
}

bool TfliteMicroModel::add_layer_callback(
  TfliteMicroLayerCallback callback,
  void *arg
)
{
  return _layer_callbacks.push_back({callback, arg}, true);
}

bool TfliteMicroModel::remove_layer_callback(
  TfliteMicroLayerCallback callback,
  void *arg
)
{
  return _layer_callbacks.remove({callback, arg});
}

TfLiteStatus TfliteMicroModel::process_section_callbacks(
  TfliteMicroSection section,
  bool is_start
)
{
  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  // The recorder section callback must always be called first when starting a section
  if(is_start && _recorder != nullptr && _recorder->flags().some_set())
  {
    _recorder->section_callback(section, true);
  }
  #endif


  TfLiteStatus retval = kTfLiteOk;
  for(auto& ctx : _section_callbacks)
  {
    auto status = ctx.callback(
      section,
      is_start,
      ctx.arg
    );
    if(status != kTfLiteOk)
    {
      retval = status;
    }
  }

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  // The recorder section callback must always be called last when ending a section
  if(!is_start && _recorder != nullptr && _recorder->flags().some_set())
  {
    _recorder->section_callback(section, false);
  }
  #endif

  return retval;
}

TfLiteStatus TfliteMicroModel::process_layer_callbacks(
  TfliteMicroSection section,
  bool is_start,
  int index,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus execution_status
)
{
  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  // The recorder layer callback must always be called first when starting a layer
  if(is_start && _recorder != nullptr && _recorder->flags().some_set())
  {
    _recorder->layer_callback(
      section,
      true,
      index,
      node_and_registration,
      execution_status
    );
  }
  #endif

  TfLiteStatus retval = execution_status;
  for(auto& ctx : _layer_callbacks)
  {
    auto status = ctx.callback(
      section,
      is_start,
      index,
      node_and_registration,
      execution_status,
      ctx.arg
    );
    if(status != kTfLiteOk)
    {
      retval = status;
    }
  }

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  // The recorder layer callback must always be called last when ending a layer
  if(!is_start && _recorder != nullptr && _recorder->flags().some_set())
  {
    _recorder->layer_callback(
      section,
      false,
      index,
      node_and_registration,
      execution_status
    );
  }
  #endif

  return retval;
}

#ifdef TFLITE_MICRO_PROFILER_ENABLED
void TfliteMicroModel::profiling_metrics_callback(
    const profiling::Profiler* profiler,
    logging::Logger* logger,
    void *arg
)
{
  auto& self = *reinterpret_cast<TfliteMicroModel*>(arg);
  #ifdef __arm__
  logger->info("CPU clock: %sHz", cpputils::format_units(ml_get_cpu_clock_frequency()));
  #endif
  for(int i = 0; i < self._n_buffers; ++i)
  {
    logger->info("Buffer%d Size: %d", i, self._buffer_sizes[i]);
  }
}
#endif // TFLITE_MICRO_PROFILER_ENABLED



} // namespace npu_toolkit