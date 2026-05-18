#include "tflite_micro_model_config.h"
#include "tflite_micro_model.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"


namespace npu_toolkit
{

const void* TfliteMicroModel::flatbuffer() const
{
  return _flatbuffer;
}

tflite::MicroInterpreter* TfliteMicroModel::interpreter() const
{
  assert(_interpreter != nullptr);
  return _interpreter;
}

const tflite::MicroOpResolver* TfliteMicroModel::ops_resolver() const
{
  assert(_ops_resolver != nullptr);
  return _ops_resolver;
}

TfLiteContext* TfliteMicroModel::tflite_context() const
{
  return (_interpreter != nullptr) ? &_interpreter->context_ : nullptr;
}

TfliteMicroModelStatus* TfliteMicroModel::status()
{
  return &_status;
}

TfliteMicroAccelerator* TfliteMicroModel::accelerator(bool allow_null) const
{
  assert(allow_null || _accelerator != nullptr);
  return _accelerator;
}

tflite::MicroAllocator* TfliteMicroModel::allocator(bool allow_null) const
{
  assert(allow_null || _allocator != nullptr);
  return _allocator;
}

bool TfliteMicroModel::set_layer_callback(
    TfliteMicroLayerCallback callback,
    void* arg
)
{
  return add_layer_callback(callback, arg);
}


#ifdef TFLITE_MICRO_PROFILER_ENABLED
profiling::Profiler* TfliteMicroModel::root_profiler() const
{
  return (_profiler != nullptr) ? _profiler->root_profiler() : nullptr;
}

TfliteMicroProfiler* TfliteMicroModel::profiler() const
{
  assert(_profiler != nullptr);
  return _profiler;
}

bool TfliteMicroModel::set_profiler_enabled(bool enable)
{
  if(_profiler == nullptr && enable)
  {
    _profiler = TfliteMicroProfiler::create();
    if(_profiler != nullptr)
    {
      _profiler->register_callbacks(this);
    }
  }
  else if(_profiler != nullptr && !enable)
  {
    _profiler->unregister_callbacks(this);
    _profiler->destroy();
    _profiler = nullptr;
  }
  return true;
}

bool TfliteMicroModel::is_profiler_enabled() const
{
  return _profiler != nullptr;
}
#endif // TFLITE_MICRO_PROFILER_ENABLED

#ifdef TFLITE_MICRO_RECORDER_ENABLED
TfliteMicroRecorder* TfliteMicroModel::recorder() const
{
  return _recorder;
}

bool TfliteMicroModel::set_recording_flags(RecordingFlags flags)
{
  if(_recorder == nullptr && flags != RecordingFlags::Disabled)
  {
    _recorder = TfliteMicroRecorder::create();
  }
  else if(_recorder != nullptr && flags == RecordingFlags::Disabled)
  {
    _recorder->destroy();
    _recorder = nullptr;
  }

  if(_recorder != nullptr)
  {
    _recorder->set_flags(flags);
  }
  return true;
}

TfliteMicroModel::RecordingFlags TfliteMicroModel::get_recording_flags() const
{
  if(_recorder != nullptr)
  {
    return _recorder->flags().value;
  }
  return TfliteMicroModel::RecordingFlags::Disabled;
}

bool TfliteMicroModel::get_recorded_data(
  const uint8_t** buffer_ptr,
  uint32_t* length_ptr
)
{
  if(_recorder != nullptr)
  {
    return _recorder->get_recorded_data(buffer_ptr, length_ptr);
  }
  return false;
}

#endif // TFLITE_MICRO_RECORDER_ENABLED


} // namespace npu_toolkit
