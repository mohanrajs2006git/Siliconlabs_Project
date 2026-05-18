#include "tflite_micro_model_config.h"

#include "ml/third_party/tflm/micro_interpreter.h"
#include "ml/third_party/flatbuffers/flatbuffers.hpp"  // from @flatbuffers
#include "ml/third_party/tflm/c_api_types.h"
#include "ml/third_party/tflm/common.h"
#include "ml/third_party/tflm/flatbuffer_utils.h"
#include "ml/third_party/tflm/memory_helpers.h"
#include "ml/third_party/tflm/micro_allocator.h"
#include "ml/third_party/tflm/micro_log.h"
#include "ml/third_party/tflm/micro_op_resolver.h"
#include "ml/third_party/tflm/micro_profiler_interface.h"
#include "ml/third_party/tflm/flatbuffer_conversions_bridge.h"
#include "ml/third_party/tflm/tls-schema_generated.h"
#include "ml/third_party/tflm/tls-schema_utils.h"

#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "tflite_micro_model.hpp"



namespace tflite
{

using namespace npu_toolkit;


static inline TfLiteStatus process_layer(
  TfLiteBridgeBuiltinDataAllocator* builtin_data_allocator,
  const MicroOpResolver& op_resolver,
  NodeAndRegistration& node_and_registration,
  const Operator* op,
  const OperatorCode* opcode,
  BuiltinOperator op_type
)
{
  TfLiteStatus status = GetRegistrationFromOpCode(
    opcode,
    op_resolver,
    &node_and_registration.registration
  );

  if (status != kTfLiteOk)
  {
    NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR(
      "Missing registration in provided OpsResolver"
    );
    return status;
  }

  const auto* registration = node_and_registration.registration;
  TFLITE_DCHECK(registration != nullptr);

  const char* custom_data = nullptr;
  size_t custom_data_size = 0;
  unsigned char* builtin_data = nullptr;

  if (op_type == BuiltinOperator_CUSTOM)
   {
    // Custom Ops may or may not have a non-null custom_options field.
    if (op->custom_options() != nullptr)
    {
      custom_data = reinterpret_cast<const char*>(op->custom_options()->data());
      custom_data_size = op->custom_options()->size();
    }
  }
  else
  {
    if (op->custom_options() != nullptr)
    {
      NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR(
        "Builtin operator with custom options"
      );
      return kTfLiteError;
    }

    TfLiteBridgeBuiltinParseFunction parser = op_resolver.GetOpDataParser(op_type);
    if (parser == nullptr)
    {
      NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR(
        "Did not find a parser"
      );
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(CallBuiltinParseFunction(
        parser, op, builtin_data_allocator, (void**)(&builtin_data)));
  }

  TfLiteIntArray* inputs_array =
      FlatBufferVectorToTfLiteTypeArray(op->inputs());
  TfLiteIntArray* outputs_array =
      FlatBufferVectorToTfLiteTypeArray(op->outputs());

  TfLiteNode* node = &node_and_registration.node;
  *node = {};
  node->inputs = inputs_array;
  node->outputs = outputs_array;
  node->builtin_data = reinterpret_cast<void*>(builtin_data);
  node->custom_initial_data = custom_data;
  node->custom_initial_data_size = custom_data_size;

  if (op->intermediates() && (op->intermediates()->size() > 0))
  {
    node->intermediates =
        FlatBufferVectorToTfLiteTypeArray(op->intermediates());
  }

  return kTfLiteOk;
}


TfLiteStatus MicroInterpreter::PrepareNodeAndRegistrationDataFromFlatbuffer() {
  TfLiteStatus retval = kTfLiteOk;
  auto& model_context = *TfliteMicroModelHelper::tflite_micro_model_context(&context_);
  auto& model = *model_context.model();

  TF_LITE_ENSURE_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Load,
    true
  ));

  for (int subgraph_idx = 0; subgraph_idx < graph_.NumSubgraphs();
       subgraph_idx++) {
    const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
    TFLITE_DCHECK(subgraph != nullptr);

    auto* opcodes = model_->operator_codes();
    TfLiteBridgeBuiltinDataAllocator* builtin_data_allocator =
        allocator_.GetBuiltinDataAllocator();
    uint32_t operators_size = NumSubgraphOperators(subgraph);

    for (size_t i = 0; i < operators_size; ++i) {
      auto& node_and_registration = graph_.GetAllocations()[subgraph_idx].node_and_registrations[i];
      const auto* op = subgraph->operators()->Get(i);
      const size_t index = op->opcode_index();

      if (index >= opcodes->size()) {
        NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR(
          "Malformed .tflite, invalid opcode index"
        );
        return kTfLiteError;
      }

      const auto* opcode = opcodes->Get(index);
      const auto op_type = GetBuiltinCode(opcode);

      model_context.set_current_layer(i, op_type);

      TF_LITE_ENSURE_STATUS(model.process_layer_callbacks(
        TfliteMicroSection::Load,
        true,
        i,
        node_and_registration
      ));

      auto status = process_layer(
        builtin_data_allocator,
        op_resolver_,
        node_and_registration,
        op,
        opcode,
        op_type
      );
      if(status != kTfLiteOk)
      {
        retval = status;
      }

      TF_LITE_ENSURE_STATUS(model.process_layer_callbacks(
        TfliteMicroSection::Load,
        false,
        i,
        node_and_registration
      ));

      model_context.clear_current_layer();
    }
  }

  TF_LITE_ENSURE_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Load,
    false
  ));

  return retval;
}



} // namespace tflite