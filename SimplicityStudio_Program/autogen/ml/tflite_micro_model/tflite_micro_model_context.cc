#include "tflite_micro_model_config.h"
#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"
#include "tflite_micro_model.hpp"

namespace npu_toolkit
{

TfliteMicroModelContext* TfliteMicroModelContext::create(
  TfLiteContext *context
)
{
  auto buffer = context->AllocatePersistentBuffer(
    context,
    sizeof(TfliteMicroModelContext)
  );

  if(buffer == nullptr)
  {
    return nullptr;
  }

  return new(buffer)TfliteMicroModelContext();
}

bool TfliteMicroModelContext::init(TfliteMicroModel* model)
{
  _model = model;
  auto micro_context = tflite::GetMicroContext(model->tflite_context());
  assert(micro_context != nullptr);
  micro_context->set_external_context(this);
  return true;
}

void TfliteMicroModelContext::deinit()
{
  auto micro_context = tflite::GetMicroContext(_model->tflite_context());
  assert(micro_context != nullptr);
  micro_context->set_external_context(nullptr);
  _model = nullptr;
}

TfliteMicroModel* TfliteMicroModelContext::model() const
{
  assert(_model != nullptr);
  return _model;
}


void TfliteMicroModelContext::set_current_layer(int index, int32_t op)
{
  current_layer_index = index;
  current_layer_opcode = (tflite::BuiltinOperator)op;
}

void TfliteMicroModelContext::clear_current_layer()
{
  current_layer_index = -1;
  current_layer_opcode = (tflite::BuiltinOperator)-1;
}

} // namespace npu_toolkit