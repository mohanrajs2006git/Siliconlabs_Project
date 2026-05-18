#include "tflite_micro_model_config.h"
#pragma once

#include <cassert>

#include "ml/utils/helpers.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"


namespace npu_toolkit
{

class TfliteMicroModel;


class TfliteMicroModelHelper
{
public:
    static const tflite::Model* flatbuffer_model(TfLiteContext* context = nullptr);
    static TfliteMicroModel* tflite_micro_model(TfLiteContext* context = nullptr);
    static TfliteMicroModelContext* tflite_micro_model_context(TfLiteContext* context = nullptr);
    static tflite::MicroAllocator* tflite_micro_model_allocator(TfLiteContext* context = nullptr);
    static tflite::BuiltinOperator current_layer_opcode(TfLiteContext* context = nullptr);
    static int current_layer_index(TfLiteContext* context = nullptr);
    static const char* current_layer_name(TfLiteContext* context = nullptr);

    static void set_active_context(TfLiteContext* context);
    static TfLiteContext* get_active_context(bool allow_null = false);


    template<typename T = uint8_t>
    static  T* allocate_persistent_buffer(
      TfLiteContext* context,
      unsigned count
    )
    {
      if(context == nullptr) return nullptr;
      return reinterpret_cast<T*>(context->AllocatePersistentBuffer(context, sizeof(T) * count));
    }

    template<typename T = uint8_t>
    static  T* allocate_planned_persistent_buffer(
      TfLiteContext* context,
      unsigned count
    )
    {
      T* retval = nullptr;
      auto allocator = tflite_micro_model_allocator(context);

      if(allocator != nullptr)
      {
        retval = reinterpret_cast<T*>(allocator->allocate_planned_persistent_buffer(count * sizeof(T)));
      }

      if(retval == nullptr)
      {
        retval = allocate_persistent_buffer<T>(context, count);
      }

      return retval;
    }


    template<typename T = uint8_t>
    static TfLiteStatus allocate_scratch_buffer(
      TfLiteContext* context,
      unsigned count,
      int *buffer_index
    )
    {
        return context->RequestScratchBufferInArena(context, sizeof(T)*count, buffer_index);
    }

    template<typename T = uint8_t>
    static T* get_scratch_buffer(TfLiteContext* context, int scratch_buffer_index)
    {
        return reinterpret_cast<T*>(context->GetScratchBuffer(context, scratch_buffer_index));
    }

    static const char* opcode_to_str(tflite::BuiltinOperator opcode);
    static const char* create_layer_name(int layer_idx, tflite::BuiltinOperator opcode);
    static bool verify_model_flatbuffer(const void* flatbuffer, int flatbuffer_length);
    static const void* get_metadata_from_tflite_flatbuffer(
      const void* tflite_flatbuffer,
      const char* tag,
      uint32_t* length = nullptr
    );
    static const void* get_metadata_from_tflite_flatbuffer(
      TfLiteContext* context,
      const char* tag,
      uint32_t* length = nullptr
    );
    static const void* get_metadata_from_tflite_flatbuffer(
      const tflite::Model *model,
      const char* tag,
      uint32_t* length = nullptr
    );
};


class TfliteMicroModelContextManager
{
public:
  TfliteMicroModelContextManager(TfLiteContext *context)
  {
    TfliteMicroModelHelper::set_active_context(context);
  }

  ~TfliteMicroModelContextManager()
  {
    TfliteMicroModelHelper::set_active_context(nullptr);
  }
};


#define NPU_TOOLKIT_ALLOCATE_PERSISTENT_BUFFER(type, count) \
  ::npu_toolkit::TfliteMicroModelHelper::allocate_persistent_buffer<type>(context, count)
#define NPU_TOOLKIT_ALLOCATE_PLANNED_PERSISTENT_BUFFER(type, count) \
  ::npu_toolkit::TfliteMicroModelHelper::allocate_planned_persistent_buffer<type>(context, count)
#define NPU_TOOLKIT_ALLOCATE_SCRATCH_BUFFER(size_bytes, scratch_buffer_index) \
  ::npu_toolkit::TfliteMicroModelHelper::allocate_scratch_buffer(context, size_bytes, scratch_buffer_index)
#define NPU_TOOLKIT_GET_SCRATCH_BUFFER(type, scratch_buffer_index) \
  ::npu_toolkit::TfliteMicroModelHelper::get_scratch_buffer<type>(context, scratch_buffer_index)

} // namespace npu_toolkit