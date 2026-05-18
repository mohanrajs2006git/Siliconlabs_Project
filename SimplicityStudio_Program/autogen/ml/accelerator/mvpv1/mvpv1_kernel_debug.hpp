#include "tflite_micro_model_config.h"
#pragma once

#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/accelerator/mvpv1/mvpv1_helpers.h"


namespace npu_toolkit
{

#ifdef TFLITE_MICRO_LOGGER_ENABLED
#if defined(MVPV1_ACCELERATOR_DUMP_REGISTERS_ENABLED)
  #define MVPV1_ACCELERATOR_DEBUG_DEFAULT_LEVEL  logging::Level::Info
#elif !defined(__arm__)
  #define MVPV1_ACCELERATOR_DUMP_REGISTERS_ENABLED
  #define MVPV1_ACCELERATOR_DEBUG_DEFAULT_LEVEL logging::Level::Disabled
#else
  #define MVPV1_ACCELERATOR_DEBUG_DEFAULT_LEVEL logging::Level::Disabled
#endif



static inline logging::Logger& mvpv1_logger()
{
  return get_logger(LoggerId::Accelerator, MVPV1_ACCELERATOR_DEBUG_DEFAULT_LEVEL);
}

#else

#undef MVPV1_ACCELERATOR_DUMP_REGISTERS_ENABLED

#endif // TFLITE_MICRO_LOGGER_ENABLED

/// @cond NO_DOXYGEN
#ifdef MVPV1_ACCELERATOR_DUMP_REGISTERS_ENABLED
  #define MVPV1_DEBUG_PRINTF(msg, ...) mvpv1_logger().info(msg, ## __VA_ARGS__)
  #define MVPV1_DEBUG_DUMP_PROGRAM(...) dump_mvpv1_program(__VA_ARGS__)
  static inline void dump_mvpv1_program(const void *prog = nullptr)
  {
    auto& logger = npu_toolkit::mvpv1_logger();

    auto vwriter = [](const char* msg, va_list args, void *arg)
    {
      auto& l = *reinterpret_cast<logging::Logger*>(arg);
      l.vwrite(logging::Info, msg, args);
    };
    mvpv1_set_dump_writer(vwriter, &logger);
    mvpv1_dump_registers(prog);
  }
#else
  #define MVPV1_DEBUG_PRINTF(...)
  #define MVPV1_DEBUG_DUMP_PROGRAM(...)
#endif
/// @endcond






} // namespace npu_toolkit