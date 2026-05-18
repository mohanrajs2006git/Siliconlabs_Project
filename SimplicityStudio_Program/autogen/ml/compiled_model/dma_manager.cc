#include "tflite_micro_model_config.h"
#include <new>
#include <cassert>
#include <algorithm>
#include <cstring>


#include "ml/compiled_model/compiled_model_dma_manager.hpp"



namespace npu_toolkit
{


CompiledModelDmaManager::CompiledModelDmaManager()
{
    memset(_channels, 0, sizeof(_channels));

    for(int i = 0; i < (int)DmaChannelType::Count; ++i)
    {
      _channels[i].hw_ch = -1;
    }
    for(int i = 0; i < MAX_DMA_HW_CH_COUNT; ++i)
    {
      _hw_ch_to_virtual_ch_map[i] = DmaChannelType::Invalid;
    }
    _hw_ch_mask = 0;
    _wait_cycles = 0;
}


void CompiledModelDmaManager::on_dma_complete(HwChannel hw_ch)
{
  const auto ch = _hw_ch_to_virtual_ch_map[hw_ch];
  if(ch == DmaChannelType::Invalid)
  {
    assert(!"Invalid channel");
    return;
  }

  auto& ch_ctx = _channels[(int)ch];
  _hw_ch_to_virtual_ch_map[hw_ch] = DmaChannelType::Invalid;

  if(ch_ctx.callback != nullptr)
  {
    ch_ctx.callback(ch_ctx.callback_cls, ch_ctx.callback_buffer_id);
  }
}

void CompiledModelDmaManager::on_dma_error(HwChannel hw_ch)
{
  assert(!"DMA error");
}

bool CompiledModelDmaManager::is_active(DmaChannelType ch) const
{
  CriticalSection critical_section;

  const auto& ch_ctx = _channels[(int)ch];
  if(ch_ctx.hw_ch == -1)
  {
    return false;
  }
  return _hw_ch_to_virtual_ch_map[ch_ctx.hw_ch] != DmaChannelType::Invalid;
}



} // namespace npu_toolkit

