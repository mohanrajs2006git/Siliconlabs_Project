#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_context.hpp"
#include "ml/compiled_model/compiled_model_context.hpp"
#include "ml/compiled_model/compiled_model.hpp"
#include "ml/accelerator/mvpv1/mvpv1_accelerator_utils.hpp"
#include "ml/accelerator/mvpv1/mvpv1_program_recorder.hpp"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_accelerator.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "tflite_micro_model.hpp"
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recorder.hpp"
#endif


namespace npu_toolkit
{

extern bool mvpv1_compiled_kernel_init(TfLiteContext *context);


bool TfliteMicroMvpv1Context::init(TfliteMicroModel* model)
{
  if(!TfliteMicroModelContext::init(model))
  {
    return false;
  }

  if(!mvpv1_compiled_kernel_init(model->tflite_context()))
  {
    return false;
  }

  return true;
}

bool TfliteMicroMvpv1Context::load()
{
  auto mvpv1_context = TfliteMicroMvpv1Context::get(_model->tflite_context());
  if(mvpv1_context->compiled_context != nullptr)
  {
    if(!mvpv1_context->compiled_context->load())
    {
      return false;
    }
  }

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  auto recorder = TfliteMicroRecorder::get();
  if(recorder != nullptr)
  {
    recorder->add_layer_callback(mvpv1_kernel_record_layer_callback);
  }
  #endif // TFLITE_MICRO_RECORDER_ENABLED

  return true;
}


} // namespace npu_toolkit