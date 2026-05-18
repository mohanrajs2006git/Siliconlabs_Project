#include "tflite_micro_model_config.h"
#include <new>
#include "ml/compiled_model/compiled_model_context.hpp"
#include "ml/compiled_model/compiled_model_cache.hpp"

namespace npu_toolkit
{


CompiledModelContext* CompiledModelContext::create(
  TfLiteContext *context,
  const void* compiled_data
)
{
  auto buffer = context->AllocatePersistentBuffer(context, sizeof(CompiledModelContext));
  if(buffer == nullptr)
  {
    return nullptr;
  }

  return new(buffer)CompiledModelContext(
    context,
    compiled_data
  );
}

CompiledModelContext::CompiledModelContext(
  TfLiteContext *context,
  const void* compiled_data
) :
  _context(context),
  _compiled_data(compiled_data),
  _compiled_cache(nullptr)
{
}

bool CompiledModelContext::init(
  PrepareLayerCallback prepare_layer_callback
)
{
  const auto compiled_model_data = CompiledModelData::create(_compiled_data);
  const auto mode = compiled_model_data->context_mode();
  const auto layer_config = compiled_model_data->layer_config_ptr();
  const auto cache_config = compiled_model_data->cache_config_ptr();

  if(!_layer_config_mgr.init(
    _context,
    layer_config,
    mode
  ))
  {
    return false;
  }

  if(cache_config != nullptr)
  {
    auto compiled_cache = CompiledModelCache::create(_context);
    if(compiled_cache == nullptr)
    {
      return false;
    }
    if(!compiled_cache->init(
      _context,
      _compiled_data,
      &_layer_config_mgr,
      prepare_layer_callback
    ))
    {
      return false;
    }

    _compiled_cache = compiled_cache;
  }

  return true;
}

bool CompiledModelContext::load()
{
  if(_compiled_cache != nullptr)
  {
    _compiled_cache->process();
  }
  return true;
}

void CompiledModelContext::deinit()
{
  if(_compiled_cache != nullptr)
  {
    _compiled_cache->deinit();
    _compiled_cache = nullptr;
  }
}

CompiledAcceleratorId CompiledModelContext::get_layer_accelerator(int index) const
{
  const auto compiled_model_data = CompiledModelData::create(_compiled_data);
  const auto layer_accelerators = compiled_model_data->layer_accelerators();
  assert(index < compiled_model_data->n_layers());
  return layer_accelerators[index];
}


bool CompiledModelContext::begin_layer()
{
  if(_compiled_cache != nullptr)
  {
    if(!_compiled_cache->begin_layer(_context))
    {
      return false;
    }
  }

  auto compiled_layer = _layer_config_mgr.current();
  assert(compiled_layer  != nullptr);

  _layer_program_count = compiled_layer->n_programs();

  return true;
}

bool CompiledModelContext::next_program(
  uint8_t* n_register_groups,
  const uint8_t** register_offsets,
  const uint32_t** register_values
)
{
  if(_compiled_cache != nullptr)
  {
    if(!_compiled_cache->next_program())
    {
      return false;
    }
  }
  else
  {
    _layer_config_mgr.next();
  }

  auto compressed_program = reinterpret_cast<const CompressedProgramConfig*>(_layer_config_mgr.current());
  assert(compressed_program != nullptr);

  *n_register_groups = compressed_program->n_register_groups();
  *register_offsets = compressed_program->register_offsets();
  *register_values = compressed_program->register_values();

  return true;
}

bool CompiledModelContext::start()
{
  if(_compiled_cache != nullptr)
  {
    return _compiled_cache->start();
  }
  else
  {
    return true;
  }
}

bool CompiledModelContext::wait()
{
  if(_compiled_cache != nullptr)
  {
    return _compiled_cache->wait();
  }
  else
  {
    return true;
  }
}


bool CompiledModelContext::release(bool is_end_of_layer)
{
  if(_compiled_cache != nullptr)
  {
    return _compiled_cache->release(is_end_of_layer);
  }
  else
  {
    return true;
  }

}

bool CompiledModelContext::end_layer()
{
  if(_compiled_cache != nullptr)
  {
    return _compiled_cache->end_layer();
  }
  else
  {
    _layer_config_mgr.next();
    return true;
  }
}


} // namespace npu_toolkit