#include "tflite_micro_model_config.h"
#pragma once

#include "ml/third_party/tflm/common.h"
#include "ml/compiled_model/compiled_model_data_manager.hpp"
#include "ml/compiled_model/compiled_model_cache_interface.hpp"


namespace npu_toolkit
{
class CompiledModelCache;

class CompiledModelContext
{
public:
  typedef bool (*PrepareLayerCallback)(
      TfLiteContext *context,
      const CacheLayerConfig& layer_config
  );


  CompiledModelContext(
    TfLiteContext *context,
    const void* compiled_data
  );

  static CompiledModelContext* create(
    TfLiteContext *context,
    const void* compiled_data
  );

  template<typename ContextType>
  static CompiledModelContext* get(TfLiteContext *context)
  {
    auto npu_context = ContextType::get(context);
    return npu_context->compiled_context;
  }
  CompiledAcceleratorId get_layer_accelerator(int index) const;

  bool init(
    PrepareLayerCallback prepare_layer_callback
  );
  bool load();
  void deinit();

  bool begin_layer();
  bool next_program(
    uint8_t* n_register_groups,
    const uint8_t** register_offsets,
    const uint32_t** register_values
  );
  bool start();
  bool wait();
  bool release(bool is_end_of_layer);
  bool end_layer();

  uint16_t n_programs() const
  {
    return _layer_program_count;
  }

  uint32_t dma_wait_cycles() const
  {
    return CompiledModelDmaManager::wait_cycles();
  }

private:
  TfLiteContext *_context = nullptr;
  const void* _compiled_data = nullptr;
  CompiledModelCache* _compiled_cache = nullptr;
  CompiledModelDataManager<CompiledLayerConfig> _layer_config_mgr;
  uint16_t _layer_program_count = 0;

  friend CompiledModelCache;
};





} // namespace npu_toolkit