#include "tflite_micro_model_config.h"
#pragma once

#include <cassert>
#include <cstring>
#include "ml/third_party/tflm/common.h"
#include "ml/compiled_model/compiled_model_interface.hpp"
#include "ml/compiled_model/compiled_model_dma_manager.hpp"
#include "ml/compiled_model/compiled_model_cache_debug.hpp"

namespace npu_toolkit
{

template<typename T>
class CompiledModelDataManager
{
public:
  using DmaChannelType = CompiledModelDmaManager::DmaChannelType;

  CompiledContextMode mode() const
  {
    return _mode;
  }

  const CompiledContainer<T>* container() const
  {
    assert(_compiled_container != nullptr);
    return _compiled_container;
  }

  bool init(
    TfLiteContext *context,
    const void* compiled_data,
    CompiledContextMode mode
  )
  {
    _mode = mode;
    _compiled_container = CompiledContainer<T>::create(compiled_data);

    if(mode == CompiledContextMode::Direct)
    {
      _current_item = _compiled_container->first_item();
    }
    else if(mode == CompiledContextMode::Buffered)
    {
      const auto container_length = _compiled_container->length_bytes();
      auto buffer = context->AllocatePersistentBuffer(context, container_length);
      if(buffer == nullptr)
      {
        return false;
      }
      memcpy(buffer, compiled_data, container_length);
      _compiled_container = CompiledContainer<T>::create(buffer);
      _current_item = _compiled_container->first_item();
    }
    else
    {
      _current_item = nullptr;
      _current_buffer_id = -1;
      _pending_buffer_count = 0;
      _release_buffer_callback_arg = nullptr;
      _release_buffer_callback = nullptr;
    }

    return true;
  }

  void deinit()
  {
    _mode = CompiledContextMode::Invalid;
    _compiled_container = nullptr;
    _current_item = nullptr;
    _current_buffer_id = -1;
    _pending_buffer_count = 0;
    _release_buffer_callback_arg = nullptr;
    _release_buffer_callback = nullptr;
  }


  bool next()
  {
    if(_mode == CompiledContextMode::Invalid)
    {
      return false;
    }

    if(_mode == CompiledContextMode::Cached)
    {
      CompiledModelDmaManager::CriticalSection critical_section;

      if(_current_item == nullptr)
      {
        return update_next_container();
      }
      else
      {
        _current_item = _current_item->next();
        if((uintptr_t)_current_item >= (uintptr_t)_compiled_container->data_end())
        {
          _current_item = nullptr;
          return update_next_container();
        }
      }

      return true;
    }
    else
    {
      _current_item = _current_item->next();
      if((uintptr_t)_current_item >= (uintptr_t)_compiled_container->data_end())
      {
        _current_item = _compiled_container->first_item();
      }

      return true;
    }
  }

  const T* current()
  {
    if(_current_item == nullptr)
    {
      next();
    }
    return _current_item;
  }

  void set_next_buffer(
    const void* buffer,
    int8_t buffer_id,
    void (*release_callback)(void* arg, int),
    void* release_callback_arg
  )
  {
    assert(_pending_buffer_count < 2);

    auto& pending_buffer = _pending_buffers[_pending_buffer_count];
    pending_buffer.buffer = buffer;
    pending_buffer.buffer_id = buffer_id;
    _pending_buffer_count++;
    _DEBUG_PRINTF(0, "set_next_buffer: buffer_id=%d", buffer_id);

    _release_buffer_callback_arg = release_callback_arg;
    _release_buffer_callback = release_callback;
  }


private:
  struct PendingBuffer
  {
    const void* buffer;
    int8_t buffer_id;
  };
  CompiledContextMode _mode = CompiledContextMode::Invalid;
  const CompiledContainer<T>* _compiled_container = nullptr;
  const T* _current_item = nullptr;

  PendingBuffer _pending_buffers[2];


  void *_release_buffer_callback_arg = nullptr;
  void (*_release_buffer_callback)(void*, int) = nullptr;

  int8_t _current_buffer_id = -1;
  int8_t _pending_buffer_count = 0;


  bool update_next_container()
  {
    bool retval;
    const int8_t prev_buffer_id = _current_buffer_id;

    if(_pending_buffer_count > 0)
    {
      auto& next_buffer = _pending_buffers[0];

      _compiled_container = CompiledContainer<T>::create(next_buffer.buffer);
      _current_item = _compiled_container->first_item();
      _current_buffer_id = next_buffer.buffer_id;
      _DEBUG_PRINTF(0,"update_next_container: using pending buffer_id=%d", _current_buffer_id);

      _pending_buffers[0] = _pending_buffers[1];
      _pending_buffer_count--;
      retval = true;
    }
    else
    {
      _DEBUG_PRINTF(0,"update_next_container: null");
      _current_buffer_id = -1;
      _compiled_container = nullptr;
      _current_item = nullptr;
      retval = false;
    }

    if(_release_buffer_callback != nullptr && prev_buffer_id != -1)
    {
      _release_buffer_callback(_release_buffer_callback_arg, prev_buffer_id);
    }

    return retval;
  }
};



} // namespace npu_toolkit