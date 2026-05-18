#include "tflite_micro_model_config.h"
#pragma once

#ifdef TFLITE_MICRO_LOGGER_ENABLED
#include "ml/utils/helpers.hpp"
#include "ml/utils/logging.hpp"

#ifndef TFLITE_MICRO_LOG_DEFAULT_LEVEL
#define TFLITE_MICRO_LOG_DEFAULT_LEVEL 1
#endif
#ifndef TFLITE_MICRO_LOG_LEVEL
#define TFLITE_MICRO_LOG_LEVEL 1
#endif


namespace npu_toolkit
{

using Logger = logging::Logger;
using LogLevel = logging::Level;

enum class LoggerId : uint8_t
{
  NpuToolkit,
  Accelerator,
  CompiledModel,
  Count
};
DEFINE_ENUM_CLASS_OPERATORS(LoggerId, uint8_t);

static inline const char* to_str(LoggerId id)
{
  switch(id)
  {
    case LoggerId::NpuToolkit: return "root";
    case LoggerId::Accelerator: return "accelerator";
    case LoggerId::CompiledModel: return "compiler";
    default: return "unknown";
  }
}

Logger& get_logger(
  LoggerId id,
  LogLevel default_level = logging::Level::Info
);

static inline Logger& get_logger()
{
  return get_logger(LoggerId::NpuToolkit, logging::Level(TFLITE_MICRO_LOG_DEFAULT_LEVEL));
}

} // namespace npu_toolkit


#if defined(TFLITE_MICRO_LOG_LEVEL) && (TFLITE_MICRO_LOG_LEVEL <= 0)
#define NPU_TOOLKIT_DEBUG(msg, ...) ::npu_toolkit::get_logger().debug(msg, ## __VA_ARGS__)
#endif
#if defined(TFLITE_MICRO_LOG_LEVEL) && (TFLITE_MICRO_LOG_LEVEL <= 1)
#define NPU_TOOLKIT_INFO(msg, ...) ::npu_toolkit::get_logger().info(msg, ## __VA_ARGS__)
#endif
#if defined(TFLITE_MICRO_LOG_LEVEL) && (TFLITE_MICRO_LOG_LEVEL <= 2)
#define NPU_TOOLKIT_WARN(msg, ...) ::npu_toolkit::get_logger().warn(msg, ## __VA_ARGS__)
#endif
#if defined(TFLITE_MICRO_LOG_LEVEL) && (TFLITE_MICRO_LOG_LEVEL <= 3)
#define NPU_TOOLKIT_ERROR(msg, ...) ::npu_toolkit::get_logger().error(msg, ## __VA_ARGS__)
#endif

#endif // TFLITE_MICRO_LOGGER_ENABLED

#ifndef NPU_TOOLKIT_DEBUG
#define NPU_TOOLKIT_DEBUG(...) (void)0
#endif
#ifndef NPU_TOOLKIT_INFO
#define NPU_TOOLKIT_INFO(...) (void)0
#endif
#ifndef NPU_TOOLKIT_WARN
#define NPU_TOOLKIT_WARN(...) (void)0
#endif
#ifndef NPU_TOOLKIT_ERROR
#define NPU_TOOLKIT_ERROR(...) (void)0
#endif

#define _NPU_TOOLKIT_PRINT_CONDITION_WARNING() NPU_TOOLKIT_WARN("Condition failed: %s:%d", __FILE__, __LINE__)
#define NPU_TOOLKIT_CHECK(condition) (condition) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()
#define NPU_TOOLKIT_CHECK_EQ(x, y) ((x) == (y)) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()
#define NPU_TOOLKIT_CHECK_NE(x, y) ((x) != (y)) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()
#define NPU_TOOLKIT_CHECK_GE(x, y) ((x) >= (y)) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()
#define NPU_TOOLKIT_CHECK_GT(x, y) ((x) >  (y)) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()
#define NPU_TOOLKIT_CHECK_LE(x, y) ((x) <= (y)) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()
#define NPU_TOOLKIT_CHECK_LT(x, y) ((x) <  (y)) ? (void)0 : _NPU_TOOLKIT_PRINT_CONDITION_WARNING()

