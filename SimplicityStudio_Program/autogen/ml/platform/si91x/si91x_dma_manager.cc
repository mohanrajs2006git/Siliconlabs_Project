#include "tflite_micro_model_config.h"
#include <cassert>
#include <cstring>

#if __has_include("sl_core.h")
#include "sl_core.h"
#else
#include "em_core.h"
#endif

#include "ml/compiled_model/compiled_model_dma_manager.hpp"

#include "sl_si91x_dma.h"

#define SL_DMA_INSTANCE 0

#define GET_CPU_CYCLES() (*((const volatile uint32_t*)0xE0001004UL)) // DWT->CYCCNT




namespace npu_toolkit
{

using DmaChannelType = CompiledModelDmaManager::DmaChannelType;
using HwChannel = CompiledModelDmaManager::HwChannel;

static volatile bool _critical_section_locked = false;
static volatile uint8_t _pending_channel_mask = 0;


CompiledModelDmaManager& CompiledModelDmaManager::instance()
{
  static CompiledModelDmaManager *default_instance = nullptr;
  static uint8_t default_instance_buffer[sizeof(CompiledModelDmaManager)] alignas(alignof(CompiledModelDmaManager));

  if(default_instance == nullptr)
  {
    default_instance = new(default_instance_buffer) CompiledModelDmaManager();
  }

  return *default_instance;
}


void on_hw_dma_complete(uint32_t hw_channel, void *data)
{
  hw_channel +=1; // Need to +1 due to bug in dma driver

  auto &self = CompiledModelDmaManager::instance();
  if(_critical_section_locked)
  {
    auto dma_type = (int)self._hw_ch_to_virtual_ch_map[hw_channel];
    auto& ch = self._channels[dma_type];
    _pending_channel_mask |= (1 << dma_type);
    return;
  }

  self._dma_irq_active = true;
  self.on_dma_complete(hw_channel);
  self._dma_irq_active = false;
}

void on_hw_dma_error(uint32_t hw_channel, void *data)
{
    auto &self = CompiledModelDmaManager::instance();
    self.on_dma_error(hw_channel+1);
}


bool CompiledModelDmaManager::allocate_ch(
  TfLiteContext* context,
  DmaChannelType ch,
  Callback callback,
  void* cls
)
{
  if(_channels[(int)ch].hw_ch != -1)
  {
    return true;
  }

  sl_dma_callback_t dma_callbacks =
  {
    on_hw_dma_complete,
    on_hw_dma_error
  };

  auto& ch_ctx = _channels[(int)ch];

  sl_dma_init_t dma_init = { SL_DMA_INSTANCE };
  uint32_t hw_ch = 0;

  if(sl_si91x_dma_init(&dma_init))
  {
    return false;
  }

  for(uint32_t ch = 1; ch < 32; ++ch)
  {
    if(sl_si91x_dma_allocate_channel(SL_DMA_INSTANCE, &ch, 0) == SL_STATUS_OK)
    {
      hw_ch = ch;
      break;
    }
  }

  if(hw_ch == 0)
  {
    return false;
  }

  if(sl_si91x_dma_register_callbacks(SL_DMA_INSTANCE, hw_ch, &dma_callbacks))
  {
    return false;
  }

  ch_ctx.hw_ch = hw_ch;

  _hw_ch_mask |= (1 << (int)ch_ctx.hw_ch);

  ch_ctx.callback = callback;
  ch_ctx.callback_cls = cls;
  ch_ctx.callback_buffer_id = -1;

  return true;
}

bool CompiledModelDmaManager::release_ch(DmaChannelType ch)
{
  if(_channels[(int)ch].hw_ch != -1)
  {
    auto& ch_ctx = _channels[(int)ch];
    sl_si91x_dma_deallocate_channel(SL_DMA_INSTANCE, ch_ctx.hw_ch);
    ch_ctx.hw_ch = -1;
    ch_ctx.callback = nullptr;
    ch_ctx.callback_cls = nullptr;
    ch_ctx.callback_buffer_id = -1;
    _hw_ch_mask &= ~(1 << (int)ch_ctx.hw_ch);
    return true;
  }

  return false;
}

bool CompiledModelDmaManager::start(
  DmaChannelType ch,
  uintptr_t src_addr,
  uintptr_t dst_addr,
  uint32_t count,
  int buffer_id
)
{
  CriticalSection critical_section;

  assert(count <= MAX_DMA_XFR_COUNT);

  auto& ch_ctx = _channels[(int)ch];
  if(ch_ctx.hw_ch == -1)
  {
    assert(!"Channel not allocated");
    return false;
  }
  else if(_hw_ch_to_virtual_ch_map[(int)ch_ctx.hw_ch] != DmaChannelType::Invalid)
  {
    assert(!"Channel is active");
    return false;
  }

  assert(count <= 2048*4);
  assert(count > 0);

  ch_ctx.callback_buffer_id = buffer_id;
  _hw_ch_to_virtual_ch_map[(int)ch_ctx.hw_ch] = ch;

  sl_dma_xfer_t xfr_cfg;
  memset(&xfr_cfg, 0, sizeof(xfr_cfg));
  xfr_cfg.dest_addr = (uint32_t*)dst_addr;
  xfr_cfg.src_addr = (uint32_t*)src_addr;
  xfr_cfg.src_inc = SL_TRANSFER_SRC_INC_32;
  xfr_cfg.dst_inc = SL_TRANSFER_DST_INC_32;
  xfr_cfg.xfer_size = SL_TRANSFER_SIZE_32;
  xfr_cfg.transfer_count = count;
  xfr_cfg.dma_mode = SL_DMA_BASIC_MODE;
  xfr_cfg.transfer_type = SL_DMA_MEMORY_TO_MEMORY;

  if(sl_si91x_dma_transfer(SL_DMA_INSTANCE, ch_ctx.hw_ch, &xfr_cfg))
  {
    return false;
  }

  return true;
}


uint32_t CompiledModelDmaManager::enter_critical()
{
  uint32_t state;
  const auto irq = (SL_DMA_INSTANCE == 0) ? UDMA0_IRQn : UDMA1_IRQn;

  NVIC_DisableIRQ(irq);
  {
    state = _critical_section_locked;
    _critical_section_locked = true;
  }
  NVIC_EnableIRQ(irq);

  return state;
}

void CompiledModelDmaManager::exit_critical(uint32_t state)
{
  uint8_t pending_channel_mask = 0;
  const auto irq = (SL_DMA_INSTANCE == 0) ? UDMA0_IRQn : UDMA1_IRQn;

  NVIC_DisableIRQ(irq);
  {
    _critical_section_locked = state;
    if(!_critical_section_locked)
    {
      pending_channel_mask = _pending_channel_mask;
      _pending_channel_mask = 0;
    }
  }
  NVIC_EnableIRQ(irq);

  if(pending_channel_mask)
  {
    for(int i = 0; i < (int)DmaChannelType::Count; ++i)
    {
      if((1 << i) & pending_channel_mask)
      {
        auto& ctx = _channels[i];
        _dma_irq_active = true;
        on_dma_complete(ctx.hw_ch);
        _dma_irq_active = false;
      }
    }
  }
}

void CompiledModelDmaManager::wait_for_interrupt()
{
  __WFE();
  uint8_t pending_channel_mask = 0;
  assert(_critical_section_locked);
  const auto irq = (SL_DMA_INSTANCE == 0) ? UDMA0_IRQn : UDMA1_IRQn;

  NVIC_DisableIRQ(irq);
  {
    pending_channel_mask = _pending_channel_mask;
    _pending_channel_mask = 0;
  }
  NVIC_EnableIRQ(irq);

  if(pending_channel_mask)
  {
    for(int i = 0; i < (int)DmaChannelType::Count; ++i)
    {
      if((1 << i) & pending_channel_mask)
      {
        auto& ctx = _channels[i];
        _dma_irq_active = true;
        on_dma_complete(ctx.hw_ch);
        _dma_irq_active = false;
      }
    }
  }
}

} // namespace npu_toolkit

