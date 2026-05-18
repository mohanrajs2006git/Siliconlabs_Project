#include "tflite_micro_model_config.h"
#include "em_device.h" // Needed for RAM_SIZE
#include "tflite_micro_model.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/tflite_micro_model/tflite_micro_compiled_allocator.hpp"
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recording_allocator.hpp"
#endif

#ifdef __arm__
#if !defined(__HEAP_SIZE)
#undef __HEAP_SIZE
extern "C" uint32_t __heap_size;
#define __HEAP_SIZE ((uint32_t)(&__heap_size))
#endif // ifndef __HEAP_SIZE
#endif // __arm__



namespace npu_toolkit
{


bool TfliteMicroModel::load(
  const void* flatbuffer,
  const tflite::MicroOpResolver& op_resolver,
  uint8_t *runtime_buffer,
  int32_t runtime_buffer_size
)
{
  uint8_t* local_buffers[1] = {runtime_buffer};
  const int32_t local_buffer_sizes[1] = {runtime_buffer_size};

  return load(flatbuffer, &op_resolver, local_buffers, local_buffer_sizes, 1);
}

bool TfliteMicroModel::load(
  const void* flatbuffer,
  const tflite::MicroOpResolver* op_resolver,
  uint8_t* const buffers[],
  const int32_t buffer_sizes[],
  int32_t n_buffers
)
{
  if(flatbuffer == nullptr)
  {
      NPU_TOOLKIT_ERROR("Null flatbuffer argument");
      return false;
  }
  if(n_buffers <= 0 || buffers == nullptr || buffer_sizes == nullptr)
  {
      NPU_TOOLKIT_ERROR("Invalid buffer argument");
      return false;
  }

  uint8_t* local_buffers[n_buffers];
  int32_t local_buffer_sizes[n_buffers];

  for(int i = 0; i < n_buffers; ++i)
  {
    local_buffers[i] = buffers[i];
    local_buffer_sizes[i] = buffer_sizes[i];
  }

  // Reset the model status
  _status.reset();

  // Load any parameters added to the flatbuffer metadata
  TfliteModelParameters::load_from_tflite_flatbuffer(flatbuffer, parameters);

  // If dynamic buffer allocation is enabled
  #ifdef TFLITE_MICRO_DYNAMIC_TENSOR_BUFFER_ALLOCATION_ENABLED
  // The first buffer in the list is considered the "runtime buffer"
  // which is also used to store persistent data and temp allocations
  uint8_t* runtime_buffer = buffers[0];
  int32_t runtime_buffer_size = buffer_sizes[0];

  // If no runtime buffer was specified,
  // then we need to allocate one now
  if(runtime_buffer == nullptr)
  {
    int32_t allocated_buffer_size = 0;

    // If the runtime_buffer_size > 0, then we attempt to allocate a buffer now.
    if(runtime_buffer_size > 0)
    {
      allocated_buffer_size = runtime_buffer_size;

      #if INTPTR_MAX == INT64_MAX
      // The buffer size embedded into the .tflite is meant for a 32-bit ARM MCU
      // If we're running on a 64-bit system then add 1MB of additional memory
      // to account for 64-bit pointer overhead
      allocated_buffer_size += 1024*1024;
      #endif

      // Allocate the buffer with the specified size
      runtime_buffer = static_cast<uint8_t*>(malloc(allocated_buffer_size));
      local_buffers[0] = runtime_buffer;
      local_buffer_sizes[0] = allocated_buffer_size;

      if(runtime_buffer == nullptr)
      {
        // If there isn't enough memory to allocate the buffer
        // then fallback to the buffer size optimization algorithm
        NPU_TOOLKIT_WARN("Failed to allocate buffer with size: %d", allocated_buffer_size);
      }
      // Load the model with the buffer
      else if(load_interpreter(
        flatbuffer,
        op_resolver,
        local_buffers,
        local_buffer_sizes,
        n_buffers
      ))
      {
        // If we successfully loaded the model with the runtime memory size from the flatbuffer
        // then we're done
      }
      else
      {
        // We failed to load the interpreter with the given buffer size
        // so clean the buffer up
        free(runtime_buffer);
        runtime_buffer = nullptr;

        // If the model has unsupported layers,
        // then just return as we cannot load it
        if(_status.is_fatal_error_detected())
        {
          _status.flush();
          unload();
          return false;
        }

        // Otherwise, if the specified buffer size was too small
        // Then fallback to the buffer size optimization algorithm

        NPU_TOOLKIT_WARN("Failed to load model with buffer of size %d", runtime_buffer_size);
      }

      // If a size was specified by the API call and we failed to load the model with it
      // then just return the error
      // NOTE: If a size was specified in the model parameters and it was too small,
      //       then we just fallback to finding the optimal size below
      if(runtime_buffer == nullptr)
      {
        unload();
        return false;
      }
    }

    // If we failed to load the model because the buffer was too small
    // then use the buffer size optimization algorithm now
    if(runtime_buffer == nullptr)
    {
      local_buffers[0] = nullptr;
      local_buffer_sizes[0] = runtime_buffer_size;

      // Find the optimal buffer size
      if(!find_optimal_buffer_size(
        flatbuffer,
        op_resolver,
        local_buffers,
        local_buffer_sizes,
        n_buffers,
        runtime_buffer_size
      ))
      {
        // On failure, just return
        if(_status.is_fatal_error_detected())
        {
          _status.flush();
        }
        else
        {
          NPU_TOOLKIT_ERROR(
            "Failed to allocate buffer for model " \
            "(due to heap memory overflow or the model is not fully supported)"
          );
        }
        unload();
        return false;
      }

      allocated_buffer_size = runtime_buffer_size;

      // Allocate the buffer
      runtime_buffer = static_cast<uint8_t*>(malloc(allocated_buffer_size));
      local_buffers[0] = runtime_buffer;
      local_buffer_sizes[0] = allocated_buffer_size;

      if(runtime_buffer == nullptr)
      {
        // If this fails, something is wrong with find_optimal_buffer_size()
        NPU_TOOLKIT_WARN("Failed to allocate buffer with size: %d", allocated_buffer_size);
        unload();
        return false;
      }
      // Load the model with the buffer
      else if(!load_interpreter(
          flatbuffer,
          op_resolver,
          local_buffers,
          local_buffer_sizes,
          n_buffers
      ))
      {
        // If this fails, something is wrong with find_optimal_buffer_size()
        NPU_TOOLKIT_WARN("Failed to allocate buffer with size: %d", allocated_buffer_size);
        unload();
        return false;
      }
    }

    // At this point, the runtime memory buffer is allocated and the model flatbuffer is loaded.
    // This model now owns the buffer
    // When the model is unloaded, the buffer will be automatically freed
    _runtime_buffer = runtime_buffer;
  }
  // Else if a pre-allocated buffer was given to this API
  else // if(runtime_buffer == nullptr)
  #endif // TFLITE_MICRO_DYNAMIC_TENSOR_BUFFER_ALLOCATION_ENABLED
  {
    // Verify the runtime_buffer_size is > 0
    if(local_buffer_sizes[0] <= 0)
    {
      NPU_TOOLKIT_ERROR("Must specify runtime_buffer_size > 0");
      unload();
      return false;
    }
    // If a runtime buffer was provided, then load the interpreter with it now
    else if(!load_interpreter(
        flatbuffer,
        op_resolver,
        local_buffers,
        local_buffer_sizes,
        n_buffers
    ))
    {
      // Return the error on failure
      if(_status.is_fatal_error_detected())
      {
        _status.flush();
      }
      else
      {
        NPU_TOOLKIT_ERROR("Failed to load model with runtime buffer size: %d", local_buffer_sizes[0]);
      }
      unload();
      return false;
    }
  } // if(runtime_buffer == nullptr)


  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  // Give the simulator references to the model and tensor buffers
  if(_accelerator != nullptr)
  {
    _accelerator->set_simulator_memory("sram", local_buffers[0], local_buffer_sizes[0]);
    _accelerator->set_simulator_memory("flash", (void*)flatbuffer, 2*1024*1024);
  }
  #endif

  return true;
}

void TfliteMicroModel::unload()
{
  // Flush any error messages before unloading
  _status.flush();

  _flatbuffer = nullptr;
  _ops_resolver = nullptr;
  _allocator = nullptr;
  _buffer_sizes = nullptr;
  _n_buffers = 0;
  parameters.unload();

  if(_accelerator != nullptr)
  {
    _accelerator->deinit(tflite_context());
    _accelerator = nullptr;
  }

  auto tflite_context = this->tflite_context();
  if(tflite_context != nullptr)
  {
    auto model_context = TfliteMicroModelHelper::tflite_micro_model_context(tflite_context);
    if(model_context != nullptr)
    {
      model_context->deinit();
    }
  }

  if(_interpreter != nullptr)
  {
    _interpreter->~MicroInterpreter();
    _interpreter = nullptr;
  }

  if(_runtime_buffer != nullptr)
  {
    free(_runtime_buffer);
    _runtime_buffer = nullptr;
  }
}

bool TfliteMicroModel::is_loaded() const
{
  return _interpreter != nullptr;
}


bool TfliteMicroModel::load_interpreter(
  const void* flatbuffer,
  const tflite::MicroOpResolver* op_resolver,
  uint8_t* buffers[],
  const int32_t buffer_sizes[],
  int32_t n_buffers
)
{
  TfliteMicroModelContext* model_context = nullptr;
  auto tflite_model = tflite::GetModel(flatbuffer);

  _ops_resolver = op_resolver;
  _flatbuffer = flatbuffer;

  // Register the accelerator that was built with this application
  _accelerator = get_registered_tflite_micro_accelerator();
  if(_accelerator == nullptr)
  {
    NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR_(_status, "No accelerator registered");
    return false;
  }

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  // (Re)initialize the recorder if necessary
  if(_recorder != nullptr)
  {
    _recorder->init();
  }
  #endif

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  if(_profiler != nullptr)
  {
    _profiler->reset();
  }
  #endif

  // Initialize the accelerator
  if(!_accelerator->init())
  {
    NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR_(_status, "Failed to initialize the accelerator");
    _accelerator = nullptr;
    return false;
  }

  // Create the buffer allocator
  if(!create_allocator(
    flatbuffer,
    buffers,
    buffer_sizes,
    n_buffers
  ))
  {
    NPU_TOOLKIT_ISSUE_MODEL_WARNING_(_status, "Failed to create allocator");
    unload();
    return false;
  }

  // Create the MicroInterpreter instance
  _interpreter = new(_interpreter_buffer)tflite::MicroInterpreter(
    tflite_model,
    *op_resolver,
    _allocator
  );

  auto context = &_interpreter->context_;
  TfliteMicroModelContextManager context_manager(context);

  _status.set_op_error_reporter(context);

  model_context = _accelerator->create_context(context);
  if(model_context == nullptr)
  {
    NPU_TOOLKIT_ISSUE_MODEL_WARNING_(_status, "Failed to create TfliteMicroModelContext");
    unload();
    return false;
  }

  // Initialize the model context
  if(!model_context->init(this))
  {
    NPU_TOOLKIT_ISSUE_MODEL_WARNING_(_status, "Failed to init model context");
    unload();
    return false;
  }

  _n_buffers = n_buffers;
  _buffer_sizes = (int32_t*)_allocator->AllocatePersistentBuffer(sizeof(int32_t)*n_buffers);
  for(int i = 0; i < n_buffers; ++i)
  {
    _buffer_sizes[i] = buffer_sizes[i];
  }

  // Allocate the model
  bool retval = true;
  if(_interpreter->AllocateTensors() == kTfLiteOk)
  {
    if(!model_context->load())
    {
      NPU_TOOLKIT_ISSUE_MODEL_WARNING_(_status, "Failed to load model context");
      unload();
      return false;
    }
  }
  else
  {
    unload();
    retval = false;
  }

  return retval;
}

bool TfliteMicroModel::create_allocator(
  const void* flatbuffer,
  uint8_t* const buffers[],
  const int32_t buffer_sizes[],
  int32_t n_buffers
)
{
  // Attempt to retrieve a compiled memory plan in the metadata of the given .tflite
  auto memory_plan = TfliteMicroCompiledAllocator::retrieve_memory_plan(flatbuffer);

  // If a compiled memory plan is available
  // then create the allocator instance
  if(memory_plan != nullptr)
  {
    auto compiled_allocator = TfliteMicroCompiledAllocator::create(
      memory_plan,
      buffers,
      buffer_sizes,
      n_buffers,
      &_status
    );

    if(compiled_allocator == nullptr)
    {
      return false;
    }

    _allocator = reinterpret_cast<tflite::MicroAllocator*>(compiled_allocator);
    return true;
  }

  // No compiled memory plan is available

  // If this build expects a compiled memory plan
  // then return the error
  #ifdef TFLITE_MICRO_OFFLINE_MEMORY_PLANNING_REQUIRED
  NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR_(_status, "Offline memory planning is required for this build");
  return false;

  #elif defined(TFLITE_MICRO_RECORDER_ENABLED)
  // If recording is enabled and only 1 buffer was provided,
  // then use the recording allocator
  if(n_buffers == 1)
  {
    _allocator = TfliteMicroNpuToolkitRecordingAllocator::create(buffers[0], buffer_sizes[0]);
    return true;
  }
  #endif // TFLITE_MICRO_RECORDER_ENABLED

  // Otherwise, allocate the default TFLM allocator
  _allocator = tflite::MicroAllocator::Create(
    buffers[0],
    buffer_sizes[0]
  );

  return true;
}


#ifdef TFLITE_MICRO_DYNAMIC_TENSOR_BUFFER_ALLOCATION_ENABLED
/**
 * Do a binary search to find the optimal runtime memory size
*/
bool TfliteMicroModel::find_optimal_buffer_size(
  const void* flatbuffer,
  const tflite::MicroOpResolver* op_resolver,
  uint8_t* buffers[],
  int32_t buffer_sizes[],
  int32_t n_buffers,
  int32_t &optimal_buffer_size
)
{
  #ifdef __arm__
  int upper_limit = __HEAP_SIZE - 8*1024;
  #else
  int upper_limit = SRAM_SIZE;
  #endif
  int lower_limit = 2048;
  int last_working_buffer_size = -1;


  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  const auto saved_profiler_enabled = is_profiler_enabled();
  set_profiler_enabled(false);
  #endif
  #ifdef TFLITE_MICRO_LOGGER_ENABLED
  const auto saved_log_level = get_logger().level();
  get_logger().level(logging::Error);
  #endif
  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  const auto saved_recording_flags = get_recording_flags();
  set_recording_flags(RecordingFlags::Disabled);
  #endif

  NPU_TOOLKIT_INFO("Searching for optimal runtime memory size ...");

  // Try to get the optimal buffer size to within 128 bytes
  while((upper_limit - lower_limit) > 128)
  {
    // Get the midpoint between the upper and lower limits
    int buffer_size = (upper_limit + lower_limit) / 2;
    buffer_size = ((buffer_size + 16 - 1) / 16) * 16; // align to 16-bytes

    // Allocate the buffer
    uint8_t* buffer = static_cast<uint8_t*>(malloc(buffer_size));
    if(buffer == nullptr)
    {
      // If we failed to malloc, then we don't have enough heap memory
      // So decrease the upper limit by 8k and try again
      // (This should only happen when we first start this algorithm)
      upper_limit -= 8*1024;
      continue;
    }

    buffer_sizes[0] = buffer_size;
    buffers[0] = buffer;

    // Try to load the model with the new buffer
    if(load_interpreter(
        flatbuffer,
        op_resolver,
        buffers,
        buffer_sizes,
        n_buffers
    ))
    {
      // The model was successfully loaded
      // Save the buffer size
      last_working_buffer_size = buffer_size;

      // And set the upper limit to the working buffer size
      // (the goal is to find the smallest possible buffer size)
      upper_limit = buffer_size;

      // Also unload the interpreter so we can load it again
      unload();
    }
    else
    {
      // Immediately break out of the loop
      // if there was a fatal error
      if(_status.is_fatal_error_detected())
      {
          break;
      }

      // Otherwise, the buffer size is too small,
      // So the new lower limit is the buffer size+1
      lower_limit = buffer_size+1;
    }

    // Free the buffer for the next iteration
    free(buffer);
  }

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  set_profiler_enabled(saved_profiler_enabled);
  #endif
  #ifdef TFLITE_MICRO_LOGGER_ENABLED
  get_logger().level(saved_log_level);
  #endif
  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  set_recording_flags(saved_recording_flags);
  #endif

  if(last_working_buffer_size == -1)
  {
    optimal_buffer_size = 0;
    // Return false if we failed to find a working buffer size
    return false;
  }

  // Add some additional memory for any padding or invoking the context.GetTensor() APIs.
  last_working_buffer_size += 256;

  NPU_TOOLKIT_INFO("Determined optimal runtime memory size to be %d", last_working_buffer_size);

  // Otherwise, we found a good buffer size
  // So return success
  optimal_buffer_size = last_working_buffer_size;
  return true;
}
#endif // TFLITE_MICRO_DYNAMIC_TENSOR_BUFFER_ALLOCATION_ENABLED



} // namespace npu_toolkit