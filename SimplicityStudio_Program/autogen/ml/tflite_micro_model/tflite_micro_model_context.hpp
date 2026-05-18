#include "tflite_micro_model_config.h"
#pragma once

#include "ml/third_party/tflm/tlcc-common.h"
#include "ml/third_party/tflm/tlc-builtin_op_data.h"
#include "ml/third_party/tflm/micro_context.h"
#include "ml/third_party/tflm/micro_allocator.h"
#include "ml/third_party/tflm/micro_interpreter_context.h"
#include "ml/utils/array_helper.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_types.hpp"


namespace npu_toolkit
{
class TfliteMicroModel;
class TfliteMicroAccelerator;
class TfliteMicroProfiler;
class TfliteMicroModelStatus;
class TfliteMicroRecorder;
class TfliteMicroAcceleratorRecorder;




class TfliteMicroModelContext
{
public:

  static TfliteMicroModelContext* create(TfLiteContext *context);
  virtual bool init(TfliteMicroModel* model);
  virtual void deinit();
  virtual bool load() { return true; }
  void set_current_layer(int index, int32_t op);
  void clear_current_layer();
  TfliteMicroModel* model() const;


  static TfliteMicroModelContext* get(TfLiteContext* context)
  {
    auto micro_context = tflite::GetMicroContext(context);
    assert(micro_context != nullptr);
    auto model_context = reinterpret_cast<TfliteMicroModelContext*>(micro_context->external_context());
    assert(model_context != nullptr);
    return model_context;
  }

  tflite::BuiltinOperator current_layer_opcode = (tflite::BuiltinOperator)(-1);
  int current_layer_index = -1;

protected:

  TfliteMicroModel* _model = nullptr;

  TfliteMicroModelContext() = default;

};




} // namespace npu_toolkit