#include "tflite_micro_model_config.h"
#pragma once

#include <cstdarg>
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/utils/string.hpp"


namespace npu_toolkit
{


#ifdef TFLITE_MICRO_LOGGER_ENABLED

#if defined(COMPILED_MODEL_CACHE_DEBUG_ENABLED)
  #define COMPILED_MODEL_CACHE_DEBUG_DEFAULT_LEVEL logging::Level::Info
#elif !defined(__arm__)
  #define COMPILED_MODEL_CACHE_DEBUG_ENABLED
  #define COMPILED_MODEL_CACHE_DEBUG_DEFAULT_LEVEL logging::Level::Disabled
#else
  #define COMPILED_MODEL_CACHE_DEBUG_DEFAULT_LEVEL logging::Level::Disabled
#endif


static inline logging::Logger& compiled_model_logger()
{
  return get_logger(LoggerId::CompiledModel, COMPILED_MODEL_CACHE_DEBUG_DEFAULT_LEVEL);
}

#else

#undef COMPILED_MODEL_CACHE_DEBUG_ENABLED

#endif // TFLITE_MICRO_LOGGER_ENABLED


#ifdef COMPILED_MODEL_CACHE_DEBUG_ENABLED
  using BitmaskString = cpputils::BitmaskString;
  #define _DEBUG_PRINTF(bits, fmt, ...) { const int _bits = bits; compiled_model_logger().info(fmt, ## __VA_ARGS__);}
  #define DEBUG_PRINTF(fmt, ...) _DEBUG_PRINTF(_config.n_buffer_groups*2, fmt, ## __VA_ARGS__)
#else
  #define _DEBUG_PRINTF(...)
  #define DEBUG_PRINTF(...)
#endif




} // namespace npu_toolkit