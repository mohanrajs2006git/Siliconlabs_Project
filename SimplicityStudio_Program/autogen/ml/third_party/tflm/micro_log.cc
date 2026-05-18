#include "tflite_micro_model_config.h"
#include "ml/third_party/tflm/micro_log.h"

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"

#include "ml/third_party/tflm/debug_log.h"


void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

int MicroSnprintf(char* buffer, size_t buf_size, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int result = MicroVsnprintf(buffer, buf_size, format, args);
  va_end(args);
  return result;
}

int MicroVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist) {
  return vsnprintf(buffer, buf_size, format, vlist);
}


void VMicroPrintf(const char* format, va_list args) {
  auto& status = npu_toolkit::TfliteMicroModelStatus::instance();
  status.vissue(false, format, args);
}