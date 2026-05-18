#include "tflite_micro_model_config.h"
/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ml/third_party/tflm/micro_interpreter_graph.h"

#include "ml/third_party/flatbuffers/flatbuffers.hpp"  // from @flatbuffers
#include "ml/third_party/tflm/common.h"
#include "ml/third_party/tflm/compatibility.h"
#include "ml/third_party/tflm/flatbuffer_utils.h"
#include "ml/third_party/tflm/memory_helpers.h"
#include "ml/third_party/tflm/micro_log.h"
#include "ml/third_party/tflm/micro_profiler.h"
#include "ml/third_party/tflm/tls-schema_generated.h"
#include "ml/third_party/tflm/single_arena_buffer_allocator.h"

#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/tflite_micro_model/tflite_micro_accelerator.hpp"
#include "tflite_micro_model.hpp"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"

#ifdef TFLITE_MICRO_PROFILER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_profiler.hpp"
#endif

#define CHECK_STATUS(func) \
{ \
  const TfLiteStatus status = (func); \
  if(status != kTfLiteOk) \
  { \
    retval = status; \
  } \
}


namespace tflite {
namespace {

using namespace npu_toolkit;

const char* OpNameFromRegistration(const TFLMRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

MicroInterpreterGraph::MicroInterpreterGraph(
    TfLiteContext* context, const Model* model, MicroAllocator* allocator,
    MicroResourceVariables* resource_variables)
    : context_(context),
      model_(model),
      allocator_(allocator),
      current_subgraph_index_(0),
      resource_variables_(resource_variables) {
  if (model != nullptr) {
    subgraphs_ = model->subgraphs();
  }
}

MicroInterpreterGraph::~MicroInterpreterGraph() {}


TfLiteStatus MicroInterpreterGraph::InitSubgraphs() {
  TfLiteStatus retval = kTfLiteOk;
  int previous_subgraph_idx = current_subgraph_index_;
  auto& model_context = *TfliteMicroModelHelper::tflite_micro_model_context(context_);
  auto& model = *model_context.model();

  CHECK_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Init,
    true
  ));

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      auto& node_and_registration = subgraph_allocations_[subgraph_idx].node_and_registrations[i];
      TfLiteNode* node =  &node_and_registration.node;
      const TFLMRegistration* registration = node_and_registration.registration;

      size_t init_data_size;
      const char* init_data;
      if (registration->builtin_code == BuiltinOperator_CUSTOM) {
        init_data = reinterpret_cast<const char*>(node->custom_initial_data);
        init_data_size = node->custom_initial_data_size;
      } else {
        init_data = reinterpret_cast<const char*>(node->builtin_data);
        init_data_size = 0;
      }

      model_context.set_current_layer(i, registration->builtin_code);

      CHECK_STATUS(model.process_layer_callbacks(
        TfliteMicroSection::Init,
        true,
        i,
        node_and_registration
      ));

      if (retval == kTfLiteOk && registration->init)
      {
        node->user_data = registration->init(context_, init_data, init_data_size);;
      }

      CHECK_STATUS(model.process_layer_callbacks(
        TfliteMicroSection::Init,
        false,
        i,
        node_and_registration
      ));

      model_context.clear_current_layer();

      if(retval != kTfLiteOk)
      {
        break;
      }
    }
  }

  CHECK_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Init,
    false
  ));

  current_subgraph_index_ = previous_subgraph_idx;
  return retval;
}

TfLiteStatus MicroInterpreterGraph::PrepareSubgraphs() {
  TfLiteStatus retval = kTfLiteOk;
  int previous_subgraph_idx = current_subgraph_index_;
  auto& model_context = *TfliteMicroModelHelper::tflite_micro_model_context(context_);
  auto& model = *model_context.model();
  auto& model_status = *model.status();

  CHECK_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Prepare,
    true
  ));

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);

    for (size_t i = 0; i < operators_size; ++i) {
      auto& node_and_registration = subgraph_allocations_[subgraph_idx].node_and_registrations[i];
      TfLiteNode* node = &node_and_registration.node;
      const TFLMRegistration* registration = node_and_registration.registration;

      model_context.set_current_layer(i, registration->builtin_code);
      NPU_TOOLKIT_DEBUG("Preparing %s", TfliteMicroModelHelper::current_layer_name());

      CHECK_STATUS(model.process_layer_callbacks(
        TfliteMicroSection::Prepare,
        true,
        i,
        node_and_registration
      ));

      if (retval == kTfLiteOk && registration->prepare != nullptr) {
        auto status = registration->prepare(context_, node);
        if(status != kTfLiteOk
        )
        {
          retval = status;
          if(!model_status.is_warning_detected())
          {
            NPU_TOOLKIT_ISSUE_MODEL_WARNING("Failed to prepare with status %d", status);
          }
        }
      }

      CHECK_STATUS(model.process_layer_callbacks(
        TfliteMicroSection::Prepare,
        false,
        i,
        node_and_registration,
        retval
      ));

      auto finish_status = allocator_->FinishPrepareNodeAllocations(/*node_id=*/i);
      if(finish_status != kTfLiteOk)
      {
        retval = finish_status;
      }

      model_context.clear_current_layer();

      if(retval != kTfLiteOk)
      {
        break;
      }
    }
  }

  CHECK_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Prepare,
    false
  ));

  current_subgraph_index_ = previous_subgraph_idx;

  return retval;
}

TfLiteStatus MicroInterpreterGraph::ResetSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->reset != nullptr) {
        registration->reset(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::FreeSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TFLMRegistration* registration = subgraph_allocations_[subgraph_idx]
                                                 .node_and_registrations[i]
                                                 .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::InvokeSubgraph(int subgraph_idx) {
  if (static_cast<size_t>(subgraph_idx) >= subgraphs_->size()) {
    MicroPrintf("Accessing subgraph %d but only %d subgraphs found",
                subgraph_idx, subgraphs_->size());
    return kTfLiteError;
  }
  TfLiteStatus retval = kTfLiteOk;
  int previous_subgraph_idx = current_subgraph_index_;
  current_subgraph_index_ = subgraph_idx;
  auto& model_context = *TfliteMicroModelHelper::tflite_micro_model_context(context_);
  auto& model = *model_context.model();
  auto& model_status = *model.status();


  #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
  const auto accelerator = model.accelerator();
  const int accelerator_loop_count = (accelerator != nullptr) ? accelerator->get_profiler_loop_count() : 1;
  for(int accelerator_loop = 0; accelerator_loop < accelerator_loop_count; ++accelerator_loop) {
    model.profiler()->set_loop_index(accelerator_loop);
  #endif

  CHECK_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Invoke,
    true
  ));

  uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
  for (size_t i = 0; i < operators_size; ++i) {
    auto& node_and_registration = subgraph_allocations_[subgraph_idx].node_and_registrations[i];
    TfLiteNode* node = &node_and_registration.node;
    const TFLMRegistration* registration = node_and_registration.registration;

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfilerInterface*>(context_->profiler));
#endif

    TFLITE_DCHECK(registration->invoke);
    model_context.set_current_layer(i, registration->builtin_code);
    NPU_TOOLKIT_DEBUG("Invoking %s", TfliteMicroModelHelper::current_layer_name());

    CHECK_STATUS(model.process_layer_callbacks(
      TfliteMicroSection::Invoke,
      true,
      i,
      node_and_registration
    ));

    if(retval == kTfLiteOk)
    {
      auto status = registration->invoke(context_, node);
      if(status != kTfLiteOk)
      {
        retval = status;
      }
    }

    if(retval != kTfLiteOk)
    {
      NPU_TOOLKIT_ISSUE_MODEL_WARNING("Failed to invoke");
    }

    CHECK_STATUS(model.process_layer_callbacks(
      TfliteMicroSection::Invoke,
      false,
      i,
      node_and_registration,
      retval
    ));

    model_context.clear_current_layer();

    // All TfLiteTensor structs used in the kernel are allocated from temp
    // memory in the allocator. This creates a chain of allocations in the
    // temp section. The call below resets the chain of allocations to
    // prepare for the next call.
    allocator_->ResetTempAllocations();

    if(retval != kTfLiteOk)
    {
      break;
    }
  }

  CHECK_STATUS(model.process_section_callbacks(
    TfliteMicroSection::Invoke,
    false
  ));

  #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
  } // for(int accelerator_loop = 0; accelerator_loop < accelerator_loop_count; ++accelerator_loop)
  #endif

  current_subgraph_index_ = previous_subgraph_idx;

  return retval;
}

TfLiteStatus MicroInterpreterGraph::ResetVariableTensors() {
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    const SubGraph* subgraph = (*subgraphs_)[subgraph_idx];
    for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
      auto* tensor = subgraph->tensors()->Get(i);
      if (tensor->is_variable()) {
        size_t buffer_size;
        TF_LITE_ENSURE_STATUS(TfLiteEvalTensorByteLength(
            &subgraph_allocations_[subgraph_idx].tensors[i], &buffer_size));

        int value = 0;
        if (tensor->type() == tflite::TensorType_INT8) {
          value = tensor->quantization()->zero_point()->Get(0);
        }
        memset(subgraph_allocations_[subgraph_idx].tensors[i].data.raw, value,
               buffer_size);
      }
    }
  }
  if (resource_variables_ != nullptr) {
    resource_variables_->ResetAll();
  }

  return kTfLiteOk;
}

int MicroInterpreterGraph::NumSubgraphs() {
  return model_->subgraphs()->size();
}

void MicroInterpreterGraph::SetSubgraphAllocations(
    SubgraphAllocations* subgraph_allocations) {
  subgraph_allocations_ = subgraph_allocations;
}

size_t MicroInterpreterGraph::NumSubgraphInputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->inputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphInput(int subgraph_idx,
                                                          int input_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->inputs()->Get(input_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroInterpreterGraph::NumSubgraphOutputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->outputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphOutput(int subgraph_idx,
                                                           int output_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->outputs()->Get(output_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

}  // namespace tflite
