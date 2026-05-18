#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>
#include "ml/third_party/tflm/tlcc-common.h"
#include "ml/third_party/tflm/micro_allocator.h"


namespace npu_toolkit
{

enum class TfliteMicroSection
{
  Load,
  Init,
  Prepare,
  Invoke,
  Count
};

static inline const char* to_str(TfliteMicroSection section)
{
  switch(section)
  {
    case TfliteMicroSection::Load: return "load";
    case TfliteMicroSection::Init: return "init";
    case TfliteMicroSection::Prepare: return "prepare";
    case TfliteMicroSection::Invoke: return "invoke";
    default: break;
  }
  return "unknown";
}

typedef TfLiteStatus (*TfliteMicroSectionCallback)(
  TfliteMicroSection section,
  bool is_start,
  void* arg
);


typedef TfLiteStatus (*TfliteMicroLayerCallback)(
  TfliteMicroSection section,
  bool is_start,
  int index,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus status,
  void* arg
);


struct TfliteMicroSectionCallbackContext
{
  TfliteMicroSectionCallback callback;
  void *arg;

  bool operator == (const TfliteMicroSectionCallbackContext& other) const
  {
    return this->callback == other.callback && this->arg == other.arg;
  }
};

struct TfliteMicroLayerCallbackContext
{
  TfliteMicroLayerCallback callback = nullptr;
  void *arg = nullptr;

  bool operator == (const TfliteMicroLayerCallbackContext& other) const
  {
    return this->callback == other.callback && this->arg == other.arg;
  }
};


} // namespace npu_toolkit