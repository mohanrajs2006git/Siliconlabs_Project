#include "tflite_micro_model_config.h"
/**
 *
 *
 * Input Buffer (External memory -> cache buffer)
 * ---------------
 * 1) Initially Empty
 *    State = Empty

 * 2) Start transfer
 *    If State = Empty
 *      State = Transferring
 *
 * 3) Transfer Complete
 *    State = Full
 *
 * 4) Wait for buffer
 *    If State = Full then ready
 *
 * 5) Release buffer
 *    State = Empty
 *
 *
 * Output Buffer (Cache buffer -> external memory)
 * ----------------
 * 1) Initially Empty
 *    State = Empty
 *
 * 2) Start transfer
 *    If State = Full
 *      State = Transferring
 *
 * 3) Transfer Complete
 *    State = Empty
 *
 * 4) Wait for buffer
 *    If State = Empty then ready
 *
 * 5) Release buffer
 *    State = Full
*/

#include "ml/third_party/tflm/compatibility.h"
#include "ml/third_party/tflm/memory_helpers.h"
#include "em_device.h"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/utils/string.hpp"
#include "ml/compiled_model/compiled_model.hpp"
#include "ml/compiled_model/compiled_model_cache.hpp"
#include "tflite_micro_model.hpp"
#include "ml/tflite_micro_model/tflite_micro_compiled_allocator.hpp"
#include "ml/compiled_model/compiled_model_context.hpp"
#include "ml/compiled_model/compiled_model_dma_manager.hpp"
#include "ml/compiled_model/compiled_model_cache_debug.hpp"



namespace npu_toolkit
{

static uintptr_t get_tflite_buffers_base_addr(TfLiteContext *context);



CompiledModelCache* CompiledModelCache::create(TfLiteContext* context)
{
    auto buffer = context->AllocatePersistentBuffer(context, sizeof(CompiledModelCache));
    return new(buffer)CompiledModelCache();
}


bool CompiledModelCache::init(
    TfLiteContext *context,
    const void* compiled_data,
    CompiledModelDataManager<CompiledLayerConfig>* layer_config_mgr,
    CompiledModelContext::PrepareLayerCallback prepare_layer_callback
)
{
    const auto compiled_model_data = CompiledModelData::create(compiled_data);
    const auto& cache_config = *CacheConfig::create(compiled_model_data->cache_config_ptr());
    const auto context_mode = compiled_model_data->context_mode();
    const auto transfer_list = compiled_model_data->cache_transfer_list_ptr();
    const auto layer_config = compiled_model_data->layer_config_ptr();

    auto model = TfliteMicroModelHelper::tflite_micro_model(context);
    auto allocator = reinterpret_cast<TfliteMicroCompiledAllocator*>(model->allocator());
    auto& dma_mgr = CompiledModelDmaManager::instance();
    #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    dma_mgr.set_process_callback([](void *arg)
    {
        auto& self = *reinterpret_cast<CompiledModelCache*>(arg);
        self.process();
    }, this);
    #endif

    memset((void*)&_config, 0, sizeof(_config));
    memset((void*)&_state, 0, sizeof(_state));

    DEBUG_PRINTF("Initializing");

    _prepare_layer_callback = prepare_layer_callback;
    _config.n_buffer_groups = cache_config.n_buffer_groups();
    assert(_config.n_buffer_groups > 0);
    _config.layer_config_buffer_group_id = -1;
    _config.xfr_list_buffer_group_id = -1;
    _config.auto_wrap_mask = 0;
    _state.layer_id = CacheTransferConfig::NULL_LAYER_ID; // Ensure the layer id is null until processing begins

    for(int group_id = 0; group_id < _config.n_buffer_groups; ++group_id)
    {
        const auto buffer_group_config = cache_config.buffer_group_config(group_id);
        auto dma_channel = DmaChannelType::Invalid;

        if(buffer_group_config.type == CacheBufferType::ModelReadyOnly)
        {
            dma_channel = DmaChannelType::NonMutableData;
            _src_base_addrs[group_id] = get_tflite_buffers_base_addr(context);
            DEBUG_PRINTF("Buffer group%d base_addr=%p (ModelReadOnly)", group_id, _src_base_addrs[group_id]);
        }
        else if(buffer_group_config.type == CacheBufferType::NonPersistent)
        {
            dma_channel = DmaChannelType::MutableData;
            _src_base_addrs[group_id] = (uintptr_t)allocator->get_non_persistent_base_addr(
                buffer_group_config.memory_region
            );
            DEBUG_PRINTF("Buffer group%d mem_region=%d base_addr=%p (NonPersistent)",
                group_id,
                buffer_group_config.memory_region,
                _src_base_addrs[group_id]
            );
        }
        else if(buffer_group_config.type == CacheBufferType::PlannedPersistent)
        {
            dma_channel = DmaChannelType::MutableData;
            _src_base_addrs[group_id] = (uintptr_t)allocator->get_planned_persistent_base_addr(
                buffer_group_config.memory_region
            );
            DEBUG_PRINTF("Buffer group%d mem_region=%d base_addr=%p (PlannedPersistent)",
                group_id,
                buffer_group_config.memory_region,
                _src_base_addrs[group_id]
            );
        }
        else if(buffer_group_config.type == CacheBufferType::LayerConfig)
        {
            dma_channel = DmaChannelType::NonMutableData;
            _config.layer_config_buffer_group_id = group_id;
            _src_base_addrs[group_id] = (uintptr_t)layer_config;
            DEBUG_PRINTF("Buffer group%d base_addr=%p (LayerConfig)",
                group_id,
                _src_base_addrs[group_id]
            );
        }
        else if(buffer_group_config.type == CacheBufferType::TransferList)
        {
            dma_channel = DmaChannelType::NonMutableData;
            _config.xfr_list_buffer_group_id = group_id;
            _src_base_addrs[group_id] = (uintptr_t)transfer_list;
            DEBUG_PRINTF("Buffer group%d base_addr=%p (TransferList)",
                group_id,
                _src_base_addrs[group_id]
            );
        }
        else
        {
            assert(!"Unknown buffer type");
            return false;
        }

        if(buffer_group_config.auto_wrap)
        {
            _config.auto_wrap_mask |= (0x3 << group_id*2);
        }

        for(int ping_pong_id = 0; ping_pong_id < 2; ++ping_pong_id)
        {
            const uint16_t buffer_id = group_id*2 + ping_pong_id;
            const auto buffer_offset = cache_config.buffer_offset(buffer_id);
            const auto cache_buffer_base_addr = allocator->get_base_addr(
                buffer_offset.memory_region
            );

            _cache_buffers[buffer_id] = cache_buffer_base_addr + buffer_offset.offset;
            DEBUG_PRINTF("Buffer%d memory_region=%d offset=%05X addr=%p",
                buffer_id,
                buffer_offset.memory_region,
                buffer_offset.offset,
                _cache_buffers[buffer_id]
            );
        }

        _group_id_to_dma_ch_map[group_id] = dma_channel;
        DEBUG_PRINTF("group_id_to_dma_ch_map[%d]=%d", group_id, (int)dma_channel);
        if(!dma_mgr.allocate_ch(
            context,
            dma_channel,
            CompiledModelCache::dma_complete_handler,
            this
        ))
        {
            return false;
        }
    }

    DEBUG_PRINTF("Buffer auto_wrap=%s state=%s",
        BitmaskString(_config.auto_wrap_mask, _bits).str,
        buffer_state_to_string()
    );

    _layer_config_mgr = layer_config_mgr;
    if(!_xfr_list_mgr.init(
        context,
        compiled_model_data->cache_transfer_list_ptr(),
        context_mode
    ))
    {
        return false;
    }

    if(context_mode == CompiledContextMode::Cached)
    {
        load_first_cached_transfer_list();
    }

    return true;
}

void CompiledModelCache::deinit()
{
    DEBUG_PRINTF("De-initialize");
    _xfr_list_mgr.deinit();
    if(_layer_config_mgr != nullptr)
    {
        _layer_config_mgr->deinit();
    }

    auto& dma_mgr = CompiledModelDmaManager::instance();
    for(int group_id = 0; group_id < _config.n_buffer_groups; ++group_id)
    {
        dma_mgr.release_ch(_group_id_to_dma_ch_map[group_id]);
        _group_id_to_dma_ch_map[group_id] = DmaChannelType::Invalid;
    }
}


bool CompiledModelCache::begin_layer(TfLiteContext *context)
{
    auto model = TfliteMicroModelHelper::tflite_micro_model(context);
    auto allocator = reinterpret_cast<TfliteMicroCompiledAllocator*>(model->allocator());

    CriticalSection critical_section;
    const CompiledLayerConfig* compiled_layer = nullptr;

    critical_section.dma_mgr.clear_wait_cycles();

    _state.layer_id = TfliteMicroModelHelper::current_layer_index(context);

    for(int n_wait_loops = 0;; ++n_wait_loops)
    {
        compiled_layer = _layer_config_mgr->current();
        if(compiled_layer != nullptr)
        {
            break;
        }
        critical_section.wait_for_interrupt();

        // Process the next DMA transfer if necesary
        process();

        #ifndef __arm__
        if(n_wait_loops > 10)
        {
            NPU_TOOLKIT_ERROR("Max number of 'begin_layer' wait loops exceeded");
            return false;
        }
        #endif // __arm__
    }

    const auto layer_config_data = compiled_layer->cache_layer_config();

    if(layer_config_data == nullptr)
    {
        _state.layer_caching_enabled = false;
        return true;
    }

    const auto& layer_config = *CacheLayerConfig::create(layer_config_data);

    uint16_t release_after_use_mask = 0;
    uint16_t manual_release_mask = 0;

    for(int i = 0; i < _config.n_buffer_groups; ++i)
    {
        const CacheBufferLifetime buffer_lifetime = layer_config.buffer_lifetime(i);

        // If the buffer is not used
        if(buffer_lifetime == CacheBufferLifetime::Unused)
        {
            // Then continue to the next buffer
            continue;
        }
        else if(buffer_lifetime == CacheBufferLifetime::ManualRelease)
        {
            manual_release_mask |= (0x03 << i*2);
        }
        else if(buffer_lifetime == CacheBufferLifetime::ReleaseAfterUse)
        {
            release_after_use_mask |= (0x03 << i*2);
        }
    }

    // Prepare the accelerator for the current layer
    if(!_prepare_layer_callback(
        context,
        layer_config
    ))
    {
        return false;
    }

    // If this layer manually releases buffers
    // Then allocate a temp buffer to hold the manual release flags
    auto manual_release_flags = manual_release_mask != 0 ? layer_config.manual_release_flags() : nullptr;
    if(manual_release_flags != nullptr)
    {
        const auto n_programs = compiled_layer->n_programs();
        const auto alloc_size = n_programs * sizeof(uint16_t);
        _state.manual_release_flags = (uint16_t*)allocator->AllocateTempBuffer(alloc_size, 16);
        assert(_state.manual_release_flags != nullptr);
        memcpy((void*)_state.manual_release_flags, layer_config.manual_release_flags(), alloc_size);
    }
    else
    {
        _state.manual_release_flags = nullptr;
    }

    _state.release_after_use_mask = release_after_use_mask;
    _state.wait_mask = layer_config.initial_wait_mask();
    _state.final_wait_mask = layer_config.final_wait_mask();
    _state.direction_mask = layer_config.buffer_direction_mask();
    _state.layer_caching_enabled = (layer_config.flags() & CacheLayerConfigFlag::DelayedStart) == CacheLayerConfigFlag::None;

    DEBUG_PRINTF("Begin %s: delayed=%d init_wait=%s fin_wait=%s dir=%s release_after_use=%s man_release=%s",
        TfliteMicroModelHelper::current_layer_name(context),
        !_state.layer_caching_enabled,
        BitmaskString(_state.wait_mask, _bits).str,
        BitmaskString(_state.final_wait_mask, _bits).str,
        BitmaskString(_state.direction_mask, _bits).str,
        BitmaskString(release_after_use_mask, _bits).str,
        BitmaskString(manual_release_mask, _bits).str
    );

    // Process the next DMA transfer if necesary
    process();

    return true;
}


bool CompiledModelCache::next_program()
{
    CriticalSection critical_section;
    _layer_config_mgr->next();

    for(int n_wait_loops = 0;; ++n_wait_loops)
    {
        auto compiled_prog = _layer_config_mgr->current();
        if(compiled_prog != nullptr)
        {
            break;
        }

        // Wait for the DMA to complete
        critical_section.wait_for_interrupt();

        // Process the next DMA transfer if necesary
        process();

        #ifndef __arm__
        if(n_wait_loops > 10)
        {
            NPU_TOOLKIT_ERROR("Max number of 'next_program' wait loops exceeded");
            return false;
        }
        #endif // __arm__
    }

    return true;
}


bool CompiledModelCache::wait()
{
    // If the current layer has not been configured for caching
    // then just return
    if(!_state.layer_caching_enabled)
    {
        return true;
    }

    const uint16_t input_mask = _state.direction_mask;
    const uint16_t manual_wait_mask = CacheLayerConfig::manual_wait_mask(_state.manual_release_flags);
    const uint16_t wait_mask = _state.wait_mask | manual_wait_mask;

    CriticalSection critical_section;

    for(int n_wait_loops = 0;; ++n_wait_loops)
    {
        uint16_t shifted_wait_mask = wait_mask;

        // For each buffer
        for(int buffer_id = 0; buffer_id < CACHE_MAX_BUFFER_COUNT; buffer_id++, shifted_wait_mask >>=1)
        {
            // If we're not waiting for anymore buffers
            // then break out of the loop.
            if(shifted_wait_mask == 0)
            {
              // Then we're ready to begin processing in the MVP
                DEBUG_PRINTF("Ready: wait=%s man_wait=%s state=%s",
                    BitmaskString(wait_mask, _bits).str,
                    BitmaskString(manual_wait_mask, _bits).str,
                    buffer_state_to_string()
                );
                return true;
            }
            // If we're not waiting for the current buffer
            // then continue to the next buffer
            else if(!(shifted_wait_mask & 0x01))
            {
                continue;
            }

            const uint16_t buffer_mask = (1 << buffer_id);
            const auto expected_wait_state =
                (buffer_mask & input_mask) ? BufferState::Full : BufferState::Empty;

            if(_state.buffer_state[buffer_id] != expected_wait_state)
            {
                break;
            }
        }

        DEBUG_PRINTF("Wait: wait=%s man_wait=%s state=%s",
            BitmaskString(wait_mask, _bits).str,
            BitmaskString(manual_wait_mask, _bits).str,
            buffer_state_to_string()
        );

        critical_section.wait_for_interrupt();

        // Kick off the next DMA if necessary
        process();

        #ifndef __arm__
        if(n_wait_loops > 10)
        {
            NPU_TOOLKIT_ERROR("Max number of cache 'wait' loops exceeded");
            return false;
        }
        #endif // __arm__
    }

    return true;
}


bool CompiledModelCache::release(bool is_end_of_layer)
{
    // If the current layer has not been configured for caching
    // then just return
    if(!_state.layer_caching_enabled)
    {
        return true;
    }

    // At this point,
    // The MVP has finished processing the cached buffers.

    const uint16_t manual_wait_mask = CacheLayerConfig::manual_wait_mask(_state.manual_release_flags);
    const uint16_t manual_release_mask = CacheLayerConfig::manual_release_mask(_state.manual_release_flags);
    const uint16_t input_mask = _state.direction_mask;
    const uint16_t release_after_use_mask = _state.release_after_use_mask;
    uint16_t shifted_wait_mask = _state.wait_mask | manual_wait_mask;
    uint16_t wait_mask = 0;

    CriticalSection critical_section;

    // For each buffer
    for(int buffer_id = 0; buffer_id < CACHE_MAX_BUFFER_COUNT; buffer_id++, shifted_wait_mask >>=1)
    {
        // If we're not waiting for anymore buffers
        // then break out of the loop.
        if(shifted_wait_mask == 0)
        {
            break;
        }
        // If we're not waiting for the current buffer
        // then continue to the next buffer
        else if(!(shifted_wait_mask & 0x01))
        {
            continue;
        }

        const uint16_t buffer_mask = (1 << buffer_id);
        const BufferState released_buffer_state =
            (buffer_mask & input_mask) ? BufferState::Empty : BufferState::Full;

        if(buffer_mask & manual_wait_mask)
        {
            if(buffer_mask & manual_release_mask)
            {
                _state.buffer_state[buffer_id] = released_buffer_state;

                DEBUG_PRINTF("Manual release: buffer=%d wait=%s state=%s",
                    buffer_id,
                    BitmaskString(wait_mask, _bits).str,
                    buffer_state_to_string()
                );
            }
        }
        // If this buffer should be released after it is used
        // or if this is the end of the layer
        else if((buffer_mask & release_after_use_mask) || is_end_of_layer)
        {
            const uint8_t group_id = buffer_id / 2;
            const uint16_t group_mask = 0x3 << group_id*2;
            const uint16_t toggled_buffer_wait_mask = ~buffer_mask & group_mask;

            _state.buffer_state[buffer_id] = released_buffer_state;

            // Toggle this buffer groups ping-pong buffer id
            wait_mask |= toggled_buffer_wait_mask;

            DEBUG_PRINTF("Release: buffer=%d wait=%s eol=%d state=%s",
                buffer_id,
                BitmaskString(wait_mask, _bits).str,
                is_end_of_layer,
                buffer_state_to_string()
            );
        }
        // Otherwise, the buffer should not be released yet
        // which means we need to wait on it as necessary
        else
        {
            wait_mask |= buffer_mask;
        }
    }

    if(_state.manual_release_flags != nullptr)
    {
        _state.manual_release_flags++;
    }

    _state.wait_mask = wait_mask;

    // Kick off the next DMA if necessary
    process();

    return true;
}

bool CompiledModelCache::end_layer()
{
    CriticalSection critical_section;
    const uint16_t wait_mask = _state.final_wait_mask;
    const uint16_t input_mask = _state.direction_mask;

    _layer_config_mgr->next();
    process();

    // If the current layer has not been configured for caching
    // then just return
    if(!_state.layer_caching_enabled || wait_mask == 0)
    {
        return true;
    }

    for(;;)
    {
        uint16_t shifted_wait_mask = wait_mask;

        // For each buffer
        for(int buffer_id = 0; buffer_id < CACHE_MAX_BUFFER_COUNT; buffer_id++, shifted_wait_mask >>=1)
        {
            // If we're not waiting for anymore buffers
            // then break out of the loop.
            if(shifted_wait_mask == 0)
            {
                DEBUG_PRINTF("EOL ready: wait=%s state=%s",
                    BitmaskString(wait_mask, _bits).str,
                    buffer_state_to_string()
                );
                return true;
            }
            // If we're not waiting for the current buffer
            // then continue to the next buffer
            else if(!(shifted_wait_mask & 0x01))
            {
                continue;
            }

            const uint16_t buffer_mask = (1 << buffer_id);
            const auto expected_wait_state =
                (buffer_mask & input_mask) ? BufferState::Full : BufferState::Empty;

            if(_state.buffer_state[buffer_id] != expected_wait_state)
            {
                break;
            }
        }

        _state.layer_id = CacheTransferConfig::NULL_LAYER_ID; // clear the layer id at the end
        DEBUG_PRINTF("EOL wait: wait=%s state=%s",
            BitmaskString(wait_mask, _bits).str,
            buffer_state_to_string()
        );

        critical_section.wait_for_interrupt();
    }

    return true;

}

bool CompiledModelCache::start()
{
    if(!_state.layer_caching_enabled)
    {
        DEBUG_PRINTF("Layer delayed caching started");
        _state.layer_caching_enabled = true;
    }
    return true;
}

void CompiledModelCache::process()
{
    CriticalSection critical_section;

    // Get the current transfer config
    const auto xfr_config = _xfr_list_mgr.current();

    // If a null value was returned then the transfer isn't ready
    if(xfr_config == nullptr)
    {
        // Just exit and we will process later
        return;
    }

    {
        const uint8_t  buffer_id            = xfr_config->buffer_id();
        const uint8_t  layer_id             = xfr_config->layer_id();
        const uint16_t buffer_mask          = (1 << buffer_id);
        const uint8_t  buffer_group_id      = buffer_id / 2;
        const auto     dma_ch               = _group_id_to_dma_ch_map[buffer_group_id];
        const auto     buffer_state         = _state.buffer_state[buffer_id];

        if(critical_section.is_active(dma_ch))
        {
            return;
        }
        else if(buffer_state == BufferState::InputTransferring ||
                buffer_state == BufferState::OutputTransferring)
        {
            return;
        }
        // If this transfer's layer_id is NOT null,
        // then it is waiting on a specific layer.
        // Just return until the layer is active
        else if(layer_id != CacheTransferConfig::NULL_LAYER_ID && layer_id != _state.layer_id)
        {
            return;
        }

        uintptr_t src_addr, dst_addr;
        const uint32_t xfr_offset           = xfr_config->offset();
        const uint32_t xfr_count            = xfr_config->count();
        const auto     direction            = xfr_config->direction();

        // If this is an input buffer
        // e.g. External memory -> cache buffer
        if(direction == CacheBufferDirection::Input)
        {
            if(_state.buffer_state[buffer_id] == BufferState::Full)
            {
                if(buffer_mask & _config.auto_wrap_mask)
                {
                    DEBUG_PRINTF("Auto-wrap buffer=%d", buffer_id);
                    _xfr_list_mgr.next();
                    _state.buffer_state[buffer_id] = BufferState::InputTransferring;
                    dma_complete_handler(this, buffer_id);
                }
                return;
            }

            // Calculate the source address for the transfer
            // This is the base address
            // plus an offset (which is calculated by the model compiler)
            assert(_src_base_addrs[buffer_group_id] != 0);
            src_addr = _src_base_addrs[buffer_group_id] + xfr_offset;

            // The destination address of the corresponding cache
            // buffer as determined by the model compiler.
            dst_addr = (uintptr_t)_cache_buffers[buffer_id];

            _state.buffer_state[buffer_id] = BufferState::InputTransferring;
        }
        // Else this is an output buffer
        // e.g. cache buffer -> External memory
        else
        {
            // If the cache buffer is empty, then there's nothing to transfer
            // So just return
            if(_state.buffer_state[buffer_id] == BufferState::Empty)
            {
                return;
            }

            src_addr = (uintptr_t)_cache_buffers[buffer_id];
            assert(_src_base_addrs[buffer_group_id] != 0);
            dst_addr = _src_base_addrs[buffer_group_id] + xfr_offset;

            _state.buffer_state[buffer_id] = BufferState::OutputTransferring;
        }

        DEBUG_PRINTF("Process: buffer=%d off=%x state=%s",
            buffer_id,
            xfr_offset,
            buffer_state_to_string()
        );

        // Increment to the next DMA transfer config
        _xfr_list_mgr.next();

        assert(src_addr % 4 == 0);
        assert(dst_addr % 4 == 0);

        DEBUG_PRINTF("Copy: src=%p dst=%p cnt=%d", src_addr, dst_addr, xfr_count);
        critical_section.dma_mgr.start(
            dma_ch,
            src_addr,
            dst_addr,
            xfr_count,
            buffer_id
        );
    }
}


void CompiledModelCache::dma_complete_handler(void *cls, int buffer_id)
{
    auto& dma_mgr = CompiledModelDmaManager::instance();
    auto& self                      = *(CompiledModelCache*)cls;
    const int      group_id         = buffer_id / 2;
    const auto     prev_state       = self._state.buffer_state[buffer_id];

    // At this point, transferred
    // - input buffers are considered full
    // - output buffers are considered empty
    self._state.buffer_state[buffer_id] =
        (prev_state == BufferState::InputTransferring) ? BufferState::Full : BufferState::Empty;
    _DEBUG_PRINTF(self._config.n_buffer_groups*2, "Complete: buffer=%d state=%s",
        buffer_id,
        self.buffer_state_to_string()
    );

    if(group_id == self._config.xfr_list_buffer_group_id)
    {
        self._xfr_list_mgr.set_next_buffer(
            self._cache_buffers[buffer_id],
            buffer_id,
            CompiledModelCache::release_context_buffer,
            cls
        );
    }
    else if(group_id == self._config.layer_config_buffer_group_id)
    {
        self._layer_config_mgr->set_next_buffer(
            self._cache_buffers[buffer_id],
            buffer_id,
            CompiledModelCache::release_context_buffer,
            cls
        );
    }

    // If this is being called from the DMA IRQ
    // then try to trigger another DMA.
    if(dma_mgr.in_irq_context())
    {
        self.process();
    }
}

void CompiledModelCache::release_context_buffer(void* cls, int buffer_id)
{
    auto& self = *(CompiledModelCache*)cls;
    const uint16_t buffer_mask = (1 << buffer_id);

    if(!(buffer_mask & self._config.auto_wrap_mask))
        {
            self._state.buffer_state[buffer_id] = BufferState::Empty;
        _DEBUG_PRINTF(self._config.n_buffer_groups*2, "Release-ctx: buffer=%d state=%s",
                buffer_id,
                self.buffer_state_to_string()
            );
        // Kick off the next DMA transfer if necesary
        self.process();
    }
}

void CompiledModelCache::load_first_cached_transfer_list()
{
    auto container = _xfr_list_mgr.container();
    const uint8_t transfer_buffer_id = _config.xfr_list_buffer_group_id *2;
    const auto first_xfr_count = (container->length_bytes() + 3) / sizeof(uint32_t);

    CompiledContainer<CacheTransferConfig>::create(
        _first_xfr_config,
        0,
        sizeof(_first_xfr_config)
    );

    CacheTransferConfig::create(
        &_first_xfr_config[4],
        0,
        first_xfr_count,
        transfer_buffer_id,
        CacheBufferDirection::Input,
        CacheTransferConfig::NULL_LAYER_ID
    );

    _xfr_list_mgr.set_next_buffer(
        _first_xfr_config,
        -1,
        nullptr,
        nullptr
    );
}

const char* CompiledModelCache::buffer_state_to_string()
{
    static char buffer[32];
    char* s = buffer;
    const int buffer_count = _config.n_buffer_groups*2;

    for(int i = buffer_count-1; i >= 0; --i)
    {
        const auto buffer_state = _state.buffer_state[i];
        *s++ = (buffer_state == BufferState::Empty) ? 'E' :
               (buffer_state == BufferState::Full)  ? 'F' : 'T';
    }
    *s = 0;

    return buffer;
}


static uintptr_t get_tflite_buffers_base_addr(TfLiteContext *context)
{
    auto flatbuffer_model = TfliteMicroModelHelper::flatbuffer_model(context);
    if(flatbuffer_model == nullptr)
    {
        assert(!"Failed to get tflite model");
        return 0;
    }
    const auto buffers_vector = flatbuffer_model->buffers();
    if(buffers_vector == nullptr)
    {
        assert(!"Invalid tflite model");
        return 0;
    }

    uintptr_t start_addr = UINTPTR_MAX;
    for(int i = 0; i < (int)buffers_vector->size(); ++i)
    {
        auto buffer = buffers_vector->Get(i);
        auto data = buffer->data();
        if(data != nullptr)
        {
            auto data_ptr =  (uintptr_t)data->Data();
            if(data_ptr < start_addr)
            {
                start_addr = data_ptr;
            }
        }
    }

    return start_addr;
}



} // namespace npu_toolkit
