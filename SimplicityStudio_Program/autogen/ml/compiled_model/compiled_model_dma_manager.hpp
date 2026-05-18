#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>
#include <functional>

#include "ml/third_party/tflm/common.h"


namespace npu_toolkit
{


class CompiledModelDmaManager
{
public:
  using IrqState = uint32_t;
  using HwChannel = int8_t;
  using Callback = void (*)(void*, int);

  enum class DmaChannelType : int8_t
  {
    NonMutableData = 0,
    MutableData = 1,
    Count = 2,
    Invalid = -1
  };

  struct CriticalSection
  {
    CriticalSection() :
      dma_mgr(CompiledModelDmaManager::instance())
    {
      state = dma_mgr.enter_critical();
    }

    ~CriticalSection()
    {
      dma_mgr.exit_critical(state);
    }

    void wait_for_interrupt()
    {
      dma_mgr.wait_for_interrupt();
    }

    bool is_active(DmaChannelType ch)
    {
      return dma_mgr.is_active(ch);
    }

    CompiledModelDmaManager& dma_mgr;
    IrqState state;
  };

  static constexpr const int MAX_DMA_HW_CH_COUNT = 16;
  static constexpr const int MAX_DMA_XFR_COUNT = (32*1024)/sizeof(uint32_t);

  static CompiledModelDmaManager& instance();

  CompiledModelDmaManager();

  bool allocate_ch(
    TfLiteContext* context,
    DmaChannelType ch,
    Callback callback,
    void* cls
  );
  bool release_ch(DmaChannelType ch);

  bool start(
    DmaChannelType ch,
    uintptr_t src_addr,
    uintptr_t dst_addr,
    uint32_t count,
    int buffer_id
  );

  bool is_active(DmaChannelType ch) const;
  bool in_irq_context() const
  {
    return _dma_irq_active;
  }

  IrqState enter_critical();
  void exit_critical(IrqState state);
  void wait_for_interrupt();

  static uint32_t wait_cycles()
  {
    const auto& self = instance();
    return self._wait_cycles;
  }

  static void clear_wait_cycles()
  {
    auto& self = instance();
    self._wait_cycles = 0;
  }

  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  // Set a callback to return a CompiledModelDmaManager instance
  // This is used by the Python wrapper
  static void set_instance_callback(
    CompiledModelDmaManager* (*callback)(void*),
    void *arg = nullptr
  );
  void set_process_callback(
    void (*callback)(void*),
    void *arg = nullptr
  );
  void (*_process_callback)(void*) = nullptr;
  void* _process_callback_arg = nullptr;
  #endif

protected:

  struct ChannelContext
  {
    void* descriptors;
    Callback callback;
    void* callback_cls;
    int callback_buffer_id;
    HwChannel hw_ch;
  };

  ChannelContext _channels[(int)DmaChannelType::Count];
  DmaChannelType _hw_ch_to_virtual_ch_map[MAX_DMA_HW_CH_COUNT];
  uint32_t _hw_ch_mask;
  uint32_t _wait_cycles;
  volatile bool _dma_irq_active;

  void on_dma_complete(HwChannel hw_ch);
  void on_dma_error(HwChannel hw_ch);

  friend void on_hw_dma_complete(uint32_t hw_channel, void *data);
  friend void on_hw_dma_error(uint32_t hw_channel, void *data);
};




} // namespace npu_toolkit