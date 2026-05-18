#include "tflite_micro_model_config.h"
#include "em_device.h"
#if __has_include("sl_core.h")
#include "sl_core.h"
#else
#include "em_core.h"
#endif


#include "ml/compiled_model/compiled_model.hpp"
#include "ml/compiled_model/compiled_model_context.hpp"
#include "ml/compiled_model/compiled_model_cache.hpp"
#include "ml/accelerator/mvpv1/mvpv1_kernel_utils.hpp"
#include "ml/accelerator/mvpv1/mvpv1_helpers.h"

#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_accelerator.hpp"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_context.hpp"
#include "ml/tflite_micro_model/tflite_micro_compiled_allocator.hpp"
#include "tflite_micro_model.hpp"


namespace npu_toolkit
{


#ifdef __arm__

#define GET_CPU_CYCLES() (*((const volatile uint32_t*)0xE0001004UL)) // DWT->CYCCNT
#define ENABLE_CPU_CYCLE_COUNTER() DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk
#define DISABLE_CPU_CYCLE_COUNTER() DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk

static uint32_t* MVP_BASE_ADDR = (uint32_t*)&MVP->PROGRAMSTATE;

#else // __arm__

static uint32_t* MVP_BASE_ADDR = nullptr;
#define ENABLE_CPU_CYCLE_COUNTER()
#define DISABLE_CPU_CYCLE_COUNTER()
#define GET_CPU_CYCLES() 0

#endif // __arm__

constexpr const uint32_t MVP_FAULT_MASK = (
  MVP_IF_LOOPFAULT | MVP_IF_BUSERRFAULT  | \
  MVP_IF_BUSALIGNFAULT | MVP_IF_ALUFAULT | \
  MVP_IF_ARRAYFAULT
);



static inline void wait_for_mvp()
{
  #ifdef __arm__
  CORE_DECLARE_IRQ_STATE;
  while (!(MVP->STATUS & MVP_STATUS_IDLE))
  {
    CORE_ENTER_CRITICAL();
    if (!(MVP->STATUS & MVP_STATUS_IDLE))
    {
      DISABLE_CPU_CYCLE_COUNTER();
      //EMU_EnterEM1();
      __WFI();
      ENABLE_CPU_CYCLE_COUNTER();
    }
    CORE_EXIT_CRITICAL();
  }
  #else
  while ((MVP->STATUS & MVP_STATUS_RUNNING))
  {
  }
  #endif
}


TfLiteStatus compiled_model_process_layer_with_accelerator(
  TfLiteContext *context,
  void (*program_callback)(void*, int),
  void* program_callback_arg
)
{
  #ifndef __arm__
  // If we're running in the simulator,
  // then we do not know the MVP peripheral addresses until runtime
  MVP_BASE_ADDR = (uint32_t*)&MVP->PROGRAMSTATE;
  #endif

  auto& accelerator = TfliteMicroMvpv1Accelerator::instance();
  auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
  auto& compiled_context = *mvpv1_context.compiled_context;

  // Ensure the MVP hardware is enabled
  MVP->EN_SET = MVP_EN_EN;
  // Ensure the MVP is not active

  wait_for_mvp();

  if(!compiled_context.begin_layer())
  {
      return kTfLiteError;
  }

  // Save the currently enabled interrupt flags
  const uint32_t saved_flags = MVP->IEN;
  // Disable all enabled interrupts
  MVP->IF_CLR = MVP->IF;
  MVP->IEN_CLR = saved_flags;
  #ifdef __arm__
  MVP->IEN_SET = MVP_IEN_PROGDONE;
  NVIC_EnableIRQ(MVP_IRQn);
  #endif

  // Declare some bookkeeping variables
  uint32_t mvp_program_cycles = 0;
  bool errors_detected = false;
  const auto n_programs = compiled_context.n_programs();

  // Iterate through each compiled MVP program used by this layer
  for(int program_index = 0; program_index < n_programs; ++program_index)
  {
    uint8_t n_register_groups;
    const uint8_t* register_offsets;
    const uint32_t* register_values;
    if(!compiled_context.next_program(
      &n_register_groups,
      &register_offsets,
      &register_values
    ))
    {
      errors_detected = true;
      break;
    }
    assert(n_register_groups == 2);

    uint32_t mvp_prog_start =  GET_CPU_CYCLES();
    auto n_mvp_registers = *register_offsets++;

    // Populate the "delta" registers for this MVP program
    while(n_mvp_registers--)
    {
      const uint8_t reg_addr_offset = *register_offsets++;
      volatile uint32_t* reg_addr = &MVP_BASE_ADDR[reg_addr_offset];
      *reg_addr = *register_values++;
    }

    // After the MVP registers are the ARRAY[]->ADDRCFG registers.
    // We must convert from a relative offset to an absolute address
    auto n_addrcfg_registers = *register_offsets++;
    while(n_addrcfg_registers--)
    {
      const uint8_t array_index = *register_offsets++;
      const uint32_t rel_addr = *register_values++;

      // If array_index = 0xFF, then this is a special case for the fused_relu programs.
      // In this case, ARRAY[1] and ARRAY[0] have the same address.
      if(array_index == 0xFF)
      {
        MVP->ARRAY[1].ADDRCFG = MVP->ARRAY[0].ADDRCFG;
      }
      // If the array_index == 0xFE, then this is a special that must be handled by
      // the the compiled kernel-specific implementation.
      // So call the callback now, which will update mvpv1_array_base_addresses.
      else if(array_index == 0xFE)
      {
        assert(mvpv1_context.update_array_base_addresses_callback != nullptr);
        mvpv1_context.update_array_base_addresses_callback(mvpv1_context);
        // Also ensure caching has started if it had a delayed start
        compiled_context.start();
      }
      else
      {
        MVP->ARRAY[array_index].ADDRCFG = mvpv1_context.array_base_addresses[array_index] + rel_addr;
      }
    }

    const uint32_t mvp_prog_end = GET_CPU_CYCLES();
    mvp_program_cycles += (mvp_prog_end - mvp_prog_start);

    // Invoke the program callback if necessary
    if(program_callback != nullptr)
    {
      program_callback(program_callback_arg, program_index);
    }

    MVPV1_DEBUG_PRINTF("------------------------------------");
    MVPV1_DEBUG_PRINTF("%s: Program %d", TfliteMicroModelHelper::current_layer_name(), accelerator.program_counter++);
    MVPV1_DEBUG_DUMP_PROGRAM();

    // Wait for this MVP program's cached input tensors to be populated.
    // Also wait for any output tensor buffers to before available
    // (This does nothing if caching is not used by this layer)
    if(!compiled_context.wait())
    {
      errors_detected = true;
      break;
    }

    // Start the MVP program execution
    // NOTE: MVP_CMD_INIT is optionally included in the compiled "delta" registers
    #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    MVP->CMD_SET = accelerator.is_simulator_enabled() ? MVP_CMD_START : MVP_CMD_HALT;
    #else
    MVP->CMD_SET = MVP_CMD_START;
    #endif // TFLITE_MICRO_SIMULATOR_ENABLED

    // Wait for the current MVP program to complete
    wait_for_mvp();

    mvpv1_increment_perfcnt_override(0, MVP->PERF[0].CNT);
    #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
    mvpv1_increment_perfcnt_override(1, MVP->PERF[1].CNT);
    #endif

    // Release any cache buffers as necessary
    // (This does nothing if caching is not used by this layer)
    if(!compiled_context.release(program_index == n_programs-1))
    {
        errors_detected = true;
        break;
    }

    // Break out of the loop if any MVP errors occurred
    const uint32_t error_flags = (MVP->IF & MVP_FAULT_MASK);
    errors_detected = error_flags != 0;
    if(errors_detected)
    {
      NPU_TOOLKIT_ERROR("MVP error: 0x%08X\n", error_flags);
      break;
    }
  } // for(int program_index = 0; program_index < n_programs; ++program_index)

  if(!errors_detected && !compiled_context.end_layer())
  {
      errors_detected = true;
  }

  #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
  mvpv1_set_custom_profiling_stat("mvp_prog_cycles", mvp_program_cycles);
  mvpv1_set_custom_profiling_stat("dma_wait_cycles", compiled_context.dma_wait_cycles());
  #endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED

  MVP->IF_CLR = MVP->IF;
  MVP->IEN_SET = saved_flags;

  if(errors_detected)
  {
      return kTfLiteError;
  }
  else
  {
      return kTfLiteOk;
  }
}



/**
 * This is called by CompiledModelCache::begin_layer()
 *
 * It configures the MVP for the current layer's cache config.
*/
static bool prepare_layer_for_accelerator(
    TfLiteContext *context,
    const CacheLayerConfig& layer_config
)
{
    auto tflm_model = TfliteMicroModelHelper::tflite_micro_model(context);
    auto& mvpv1_context = *TfliteMicroMvpv1Context::get(context);
    auto allocator = reinterpret_cast<TfliteMicroCompiledAllocator*>(tflm_model->allocator());
    uintptr_t* base_addresses = (mvpv1_context.update_array_base_addresses_callback == nullptr) ?
      mvpv1_context.array_base_addresses :
      mvpv1_context.array_base_addresses2;

    const int n_mvp_array_ids = layer_config.n_mvp_array_ids();
    for(int i = 0; i < n_mvp_array_ids; ++i)
    {
        const auto mvp_array_id = layer_config.mvp_array_id(i);
        const auto mvp_array_memory_region = layer_config.mvp_array_memory_region(i);
        const auto cache_buffer_base_addr = (uintptr_t)allocator->get_base_addr(mvp_array_memory_region);

        // The array base offset is the base address of the cache RAM buffer
        // minus the arbitrary offset of CACHE_BUFFER_ADDR_OFFSET.
        // The model compiler adds this offset to each MVP program's ARRAY.ADDRCFG
        // to ensures there are no negative values in the ARRAY.ADDRCFG.
        // We subtract this offset from the ARRAYBASE to account for this.
        base_addresses[mvp_array_id] = cache_buffer_base_addr - CACHE_BUFFER_ADDR_OFFSET;
        MVPV1_DEBUG_PRINTF("Configuring MVP->ARRAYBASE[%d]=%p (region=%d, addr=%p)",
            mvp_array_id,
            (void*)mvpv1_context.array_base_addresses2[mvp_array_id],
            mvp_array_memory_region,
            cache_buffer_base_addr
        );
    }
    return true;
}


/**
 * This is called during TfliteMicroNpuContext::init()
 * It initializes the CompiledModelCache
*/
bool mvpv1_compiled_kernel_init(
    TfLiteContext *context
)
{
  auto mvp_context = TfliteMicroMvpv1Context::get(context);

  const void* compiled_data = compiled_model_retrieve_compilation_data(context);

  if(compiled_data == nullptr)
  {
    return true;
  }

  auto compiled_context = CompiledModelContext::create(
    context,
    compiled_data
  );
  if(compiled_context == nullptr)
  {
    return false;
  }

  if(!compiled_context->init(prepare_layer_for_accelerator))
  {
    return false;
  }

  mvp_context->compiled_context = compiled_context;

  return true;
}


} // namespace npu_toolkit


#if defined(__arm__) && !defined(MVPV1_ACCELERATOR_GSDK_KERNELS_ENABLED)

#if defined(SI917)
// If this is a 917 build
#define MVP_IRQHandler IRQ062_Handler
#endif // SI917

extern "C" void MVP_IRQHandler()
{
  MVP->IF_CLR = MVP->IF;
}

#endif // __arm__