#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>
#include <cassert>

#include "ml/third_party/tflm/common.h"
#include "ml/compiled_model/compiled_model_cache_interface.hpp"
#include "ml/compiled_model/compiled_model_data_manager.hpp"
#include "ml/compiled_model/compiled_model_dma_manager.hpp"
#include "ml/compiled_model/compiled_model_context.hpp"

namespace npu_toolkit
{

/**
 * This manages the caching of an ML model's tensors during
 * inference. The cacheable data includes:
 * - Weights/filters tensor
 * - Bias tensor
 * - Input tensor
 * - Output tensor
 * - Persistent data - data that is calculated once
 *     when the model is loaded and persists for the lifetime of the model.
 *     This data is only read during inference.
*/
class CompiledModelCache
{

public:
    using PrepareLayerCallback = CompiledModelContext::PrepareLayerCallback;
    using DmaChannelType = CompiledModelDmaManager::DmaChannelType;

    static CompiledModelCache* create(TfLiteContext* context);

    /**
     * Initialize the cache for the current model.
     * This is invoked before the first inference
     * by the first layer of the model that has compilation data.
     * The layer's compilation data should contain a "transfer list"
     * which is a list of offsets and lengths used to transfer
     * data from external memory (e.g. flash, PSRAM)
     * to an SRAM cache buffer. The transfer list contains
     * all the necessary transfers info for all the layers of the model
     * (i.e. for one inference)
    */
    bool init(
        TfLiteContext *context,
        const void* compiled_data,
        CompiledModelDataManager<CompiledLayerConfig>* layer_config_mgr,
        PrepareLayerCallback prepare_layer_callback
    );
    void deinit();


    /**
     * Prepare the current layer for caching.
     * If the current layer uses caching, then it should
     * contain a "buffer config". This indicates which buffers
     * are used by the layer and how the buffers should be released
     * after they are used by an MVP program.
    */
    bool begin_layer(TfLiteContext *context);
    bool next_program();
    bool end_layer();
    bool start();

    /**
     * Wait for the necessary buffers:
     * Input tensors -> Transferred from external to cache buffer
     * Output tensors -> Buffer to be allocated
     *
     * The buffer(s) are determined by the "buffer_config"
     * given to the @ref prepare() API.
    */
    bool wait();

    /**
     * Release any cache buffers that are no longer needed.
     * The buffer(s) are determined by the "buffer_config"
     * given to the @ref prepare() API.
    */
    bool release(bool is_end_of_layer);

    /**
     * Process the next DMA transfer configuration (if available).
     * This is invoked by @ref prepare() and @ref on_dma_complete().
    */
    void process();

private:
    using CriticalSection = CompiledModelDmaManager::CriticalSection;

    /**
     * This contains pointers to each of the cache buffers
    */
    uint8_t* _cache_buffers[CACHE_MAX_BUFFER_COUNT] = { nullptr };
    uintptr_t _src_base_addrs[CACHE_MAX_BUFFER_GROUP_COUNT] = { 0 };

    enum class BufferState : uint8_t
    {
        Empty = 0,
        Full = 1,
        InputTransferring = 2, // External memory -> cache buffer
        OutputTransferring = 3, // Cache buffer -> External memory (e.g. PSRAM)
    };

    struct
    {
        /**
         *
        */
        uint16_t auto_wrap_mask = 0;
        uint8_t n_buffer_groups = 0;
        int8_t xfr_list_buffer_group_id;
        int8_t layer_config_buffer_group_id;
    } _config;

    struct
    {
        volatile BufferState buffer_state[CACHE_MAX_BUFFER_COUNT] = { BufferState::Empty };
        const uint16_t* manual_release_flags = nullptr;
        uint16_t wait_mask = 0;
        uint16_t direction_mask = 0;
        uint16_t release_after_use_mask = 0;
        uint16_t final_wait_mask = 0;
        uint8_t layer_id = 0;
        bool layer_caching_enabled = false;
    } _state;


    CompiledModelDataManager<CacheTransferConfig> _xfr_list_mgr;
    CompiledModelDataManager<CompiledLayerConfig>* _layer_config_mgr = nullptr;
    uint8_t _first_xfr_config[
        CompiledContainer<CacheTransferConfig>::HEADER_LENGTH_BYTES +
        CacheTransferConfig::LENGTH_BYTES
    ];
    DmaChannelType _group_id_to_dma_ch_map[CACHE_MAX_BUFFER_GROUP_COUNT] = { DmaChannelType::Invalid };

    PrepareLayerCallback _prepare_layer_callback = nullptr;

    CompiledModelCache() = default;

    /**
     * This is an internal function invoked by the DMA IRQ handler.
    */
    static void dma_complete_handler(void *cls, int buffer_id);

    static void release_context_buffer(void* cls, int buffer_id);

    void load_first_cached_transfer_list();


    const char* buffer_state_to_string();
};


} // namespace npu_toolkit