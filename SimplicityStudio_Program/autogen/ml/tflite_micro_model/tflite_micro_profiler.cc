#include "tflite_micro_model_config.h"
#ifdef TFLITE_MICRO_PROFILER_ENABLED

#include <cstdlib>
#include <new>

#include "ml/tflite_micro_model/tflite_micro_profiler.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/tflite_micro_model/tflite_micro_accelerator.hpp"
#include "tflite_micro_model.hpp"


namespace npu_toolkit
{

TfliteMicroProfiler* TfliteMicroProfiler::create()
{
  void* instance_buffer = malloc(sizeof(TfliteMicroProfiler));
  if(instance_buffer == nullptr)
  {
    assert(!"Failed to malloc TfliteMicroProfiler");
    return nullptr;
  }

  auto instance = new(instance_buffer) TfliteMicroProfiler();

  return instance;
}

void TfliteMicroProfiler::register_callbacks(TfliteMicroModel* model)
{
  model->add_section_callback(TfliteMicroProfiler::_section_callback, this);
  model->add_layer_callback(TfliteMicroProfiler::_layer_callback, this);
}

void TfliteMicroProfiler::unregister_callbacks(TfliteMicroModel* model)
{
  model->remove_section_callback(TfliteMicroProfiler::_section_callback, this);
  model->remove_layer_callback(TfliteMicroProfiler::_layer_callback, this);
}

void TfliteMicroProfiler::reset()
{
  if(_inference_profiler != nullptr)
  {
    profiling::reset(_inference_profiler, true);
  }
}

void TfliteMicroProfiler::destroy()
{
  if(_inference_profiler != nullptr)
  {
    // destroying the inference profiler and all its children profilers
    profiling::destroy(_inference_profiler);
    _inference_profiler = nullptr;
  }
  if(_layer_profilers != nullptr)
  {
    free(_layer_profilers);
    _layer_profilers = nullptr;
  }
  free(this);
}


bool TfliteMicroProfiler::prepare_model()
{
  if(!profiling::create("Inference", _inference_profiler))
  {
    return false;
  }
  _inference_profiler->flags(profiling::Flag::ReportTotalChildrenCycles|profiling::Flag::ReportsFreeRunningCpuCycles);

  _layer_profilers = static_cast<profiling::Profiler**>(malloc(sizeof(profiling::Profiler*)*_n_layers));
  if(_layer_profilers == nullptr)
  {
    profiling::destroy(_inference_profiler);
    _inference_profiler = nullptr;
  }

  return true;
}

void TfliteMicroProfiler::register_profiler(
  int layer_index,
  tflite::BuiltinOperator opcode,
  const tflite::NodeAndRegistration& node_and_registration
)
{
  if(_inference_profiler == nullptr)
  {
    return;
  }

  profiling::Profiler* profiler;
  profiling::create(
    TfliteMicroModelHelper::create_layer_name(layer_index, opcode),
    profiler,
    _inference_profiler
  );

  if(profiler == nullptr)
  {
    return;
  }

  profiler->flags().set(profiling::Flag::TimeMeasuredBetweenStartAndStop);
  _layer_profilers[layer_index] = profiler;
  calculate_op_metrics(
    TfliteMicroModelHelper::get_active_context(),
    node_and_registration,
    profiler->metrics()
  );
}


void TfliteMicroProfiler::start_model()
{
  if(_inference_profiler != nullptr && _loop_index <= 0)
  {
    _inference_profiler->start();
  }

  auto accelerator = get_registered_tflite_micro_accelerator();
  accelerator->start_model_profiler(_loop_index);
}


void TfliteMicroProfiler::stop_model()
{
  auto accelerator = get_registered_tflite_micro_accelerator();
  accelerator->stop_model_profiler(_loop_index);

  if(_inference_profiler != nullptr && _loop_index <= 0)
  {
    _inference_profiler->stop();
  }
}


void TfliteMicroProfiler::start_layer(int index)
{
  if(_layer_profilers != nullptr)
  {
    _layer_profilers[index]->start();
  }

  auto accelerator = get_registered_tflite_micro_accelerator();
  auto layer_profiler = (_layer_profilers != nullptr) ? _layer_profilers[index] : nullptr;
  accelerator->start_layer_profiler(index, layer_profiler);
}

void TfliteMicroProfiler::stop_layer(int index)
{
  if(_layer_profilers != nullptr)
  {
    _layer_profilers[index]->stop();
  }

  auto accelerator = get_registered_tflite_micro_accelerator();
  auto layer_profiler = (_layer_profilers != nullptr) ? _layer_profilers[index] : nullptr;
  accelerator->stop_layer_profiler(index, layer_profiler);
}

void TfliteMicroProfiler::set_loop_index(int index)
{
  _loop_index = index;
}

TfLiteStatus TfliteMicroProfiler::_section_callback(
  TfliteMicroSection section,
  bool is_start,
  void* arg
)
{
  auto& self = *reinterpret_cast<TfliteMicroProfiler*>(arg);

  if(section == TfliteMicroSection::Prepare && is_start)
  {
    self.prepare_model();
  }
  else if(section == TfliteMicroSection::Invoke)
  {
    if(is_start)
    {
      self.start_model();
    }
    else
    {
      self.stop_model();
    }
  }

  return kTfLiteOk;
}

TfLiteStatus TfliteMicroProfiler::_layer_callback(
  TfliteMicroSection section,
  bool is_start,
  int index,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus status,
  void* arg
)
{
  auto& self = *reinterpret_cast<TfliteMicroProfiler*>(arg);

  if(section == TfliteMicroSection::Init && is_start)
  {
    self._n_layers++;
  }
  else if(section == TfliteMicroSection::Prepare && is_start)
  {
    self.register_profiler(
      index,
      TfliteMicroModelHelper::current_layer_opcode(),
      node_and_registration
    );
  }
  else if(section == TfliteMicroSection::Invoke)
  {
    if(is_start)
    {
      self.start_layer(index);
    }
    else
    {
      self.stop_layer(index);
    }
  }
  return status;
}

} // namespace npu_toolkit

#endif // #ifdef TFLITE_MICRO_PROFILER_ENABLED