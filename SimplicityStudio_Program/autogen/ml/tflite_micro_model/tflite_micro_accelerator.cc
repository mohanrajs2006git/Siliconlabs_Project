#include "tflite_micro_model_config.h"
#include "ml/tflite_micro_model/tflite_micro_accelerator.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "tflite_micro_model.hpp"


namespace npu_toolkit
{


#ifndef NPU_TOOLKIT_DLL_IMPORT
static TfliteMicroAccelerator* _registered_accelerator = nullptr;
extern TfliteMicroAccelerator* register_tflite_micro_accelerator(TfliteMicroAccelerator* accelerator)
{
    _registered_accelerator = accelerator;
    return accelerator;
}

extern TfliteMicroAccelerator* get_registered_tflite_micro_accelerator()
{
    return _registered_accelerator;
}
#endif // NPU_TOOLKIT_DLL_IMPORT


#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
void TfliteMicroAccelerator::set_layer_disable_filters(const std::vector<std::string>* filters)
{
  _layer_disable_filters = filters;
}

bool TfliteMicroAccelerator::is_current_layer_disabled()
{
  if(_layer_disable_filters == nullptr || _layer_disable_filters->size() == 0)
  {
    return false;
  }

  char index_str[64];
  const char* current_layer_name = TfliteMicroModelHelper::current_layer_name();
  const auto current_layer_opcode = TfliteMicroModelHelper::current_layer_opcode(
    TfliteMicroModelHelper::get_active_context()
  );
  const auto current_layer_opcode_str = TfliteMicroModelHelper::opcode_to_str(current_layer_opcode);

  const int current_layer_index = TfliteMicroModelHelper::current_layer_index(
    TfliteMicroModelHelper::get_active_context()
  );

  snprintf(index_str, sizeof(index_str), "op%d", current_layer_index);

  for(auto& filter : *_layer_disable_filters)
  {
    const char* filter_str = filter.c_str();

    if(strcasecmp(filter_str, current_layer_name) == 0)
    {
      return true;
    }
    if(strcasecmp(filter_str, current_layer_opcode_str) == 0)
    {
      return true;
    }
    if(strcasecmp(filter_str, index_str) == 0)
    {
      return true;
    }
  }

  return false;
}

bool TfliteMicroAccelerator::get_current_layer_disabled(TfLiteContext* context)
{
  auto model_context = TfliteMicroModelContext::get(context);
  auto model = model_context->model();
  auto accelerator = model->accelerator(true);
  if(accelerator == nullptr)
  {
    return false;
  }
  return accelerator->is_current_layer_disabled();
}

#endif // TFLITE_MICRO_SIMULATOR_ENABLED


} // namespace npu_toolkit