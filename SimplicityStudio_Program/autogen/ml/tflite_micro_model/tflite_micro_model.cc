#include "tflite_micro_model_config.h"
#include <new>
#include <cstdlib>

#include "ml/third_party/tflm/tls-schema_generated.h"
#include "ml/third_party/tflm/micro_arena_constants.h"
#include "ml/third_party/tflm/version.h"
#include "tflite_micro_model.hpp"
#include "ml/tflite_micro_model/tflite_micro_utils.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"

#ifdef NPU_TOOLKIT_WRAPPER_BUILD_ENABLED
#include "tflite_micro_python.hpp"
#endif



namespace npu_toolkit
{

TfliteMicroModel::TfliteMicroModel()
{
  #ifdef NPU_TOOLKIT_WRAPPER_BUILD_ENABLED
  TfliteMicroPython::init(this);
  #endif
  _status.init(this);
}


TfliteMicroModel::~TfliteMicroModel()
{
  unload();

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  // We want the recorder to persist even after
  // the model us unloaded, so we must destroy it in the destructor
  if(_recorder != nullptr)
  {
    _recorder->destroy();
    _recorder = nullptr;
  }
  #endif

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  if(_profiler != nullptr)
  {
    _profiler->destroy();
    _profiler = nullptr;
  }
  #endif

  #ifdef NPU_TOOLKIT_WRAPPER_BUILD_ENABLED
  TfliteMicroPython::deinit(this);
  #endif

}


bool TfliteMicroModel::reset()
{
  if(!is_loaded())
  {
    return false;
  }

  return _interpreter->Reset() == kTfLiteOk;
}

bool TfliteMicroModel::invoke()
{
  bool retval;
  TfliteMicroModelContextManager context_manager(tflite_context());
  auto accelerator = get_registered_tflite_micro_accelerator();

  if(!is_loaded())
  {
    NPU_TOOLKIT_ERROR("Model not loaded");
    return false;
  }

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  if(_profiler != nullptr)
  {
    _profiler->reset();
  }
  #endif

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  retval = accelerator->invoke_simulator([this]() -> bool
  {
    return _interpreter->Invoke() == kTfLiteOk;
  });

  #else // TFLITE_MICRO_SIMULATOR_ENABLED
  retval = (_interpreter->Invoke() == kTfLiteOk);
  #endif // TFLITE_MICRO_SIMULATOR_ENABLED

  // Ensure the status has been flushed
  _status.flush();

  return retval;
}

unsigned TfliteMicroModel::n_inputs() const
{
  if(!is_loaded())
  {
    NPU_TOOLKIT_ERROR("Model not loaded");
    return false;
  }

  return _interpreter->inputs_size();
}

TfliteTensorView* TfliteMicroModel::input(unsigned index) const
{
  if(!is_loaded())
  {
    NPU_TOOLKIT_ERROR("Model not loaded");
    return nullptr;
  }

  return reinterpret_cast<TfliteTensorView*>(_interpreter->input(index));
}

unsigned TfliteMicroModel::n_outputs() const
{
  if(!is_loaded())
  {
    NPU_TOOLKIT_ERROR("Model not loaded");
    return false;
  }

  return _interpreter->outputs_size();
}

TfliteTensorView* TfliteMicroModel::output(unsigned index) const
{
  if(!is_loaded())
  {
    NPU_TOOLKIT_ERROR("Model not loaded");
    return nullptr;
  }

  return reinterpret_cast<TfliteTensorView*>(_interpreter->output(index));
}

const void* TfliteMicroModel::find_metadata(const char* tag, uint32_t* length) const
{
  return TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(
    _flatbuffer,
    tag,
    length
  );
}

const char* TfliteMicroModel::tflite_micro_version()
{
  return TFLITE_MICRO_VERSION;
}


} // namespace npu_toolkit

