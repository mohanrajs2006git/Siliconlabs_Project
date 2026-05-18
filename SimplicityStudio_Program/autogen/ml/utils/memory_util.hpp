#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>


namespace npu_toolkit
{

static inline uint16_t bytes_to_uint16(const uint8_t* bytes)
{
  return (((uint16_t)bytes[0]) << 0) | (((uint16_t)bytes[1]) << 8);
}

static inline uint32_t bytes_to_uint24(const uint8_t* bytes)
{
  return (((uint16_t)bytes[0]) << 0) | (((uint16_t)bytes[1]) << 8) | (((uint16_t)bytes[2]) << 16);
}


static inline uint32_t bytes_to_uint32(const uint8_t* bytes)
{
  return (((uint32_t)bytes[0]) << 0)  | (((uint32_t)bytes[1]) << 8) |
         (((uint32_t)bytes[2]) << 16) | (((uint32_t)bytes[3]) << 24);
}

template<typename T>
T align_up_uint32(T v)
{
  return ((v + 3) >> 2) << 2;
}

template<typename T>
T align_up_uint16(T v)
{
  return ((v + 1) >> 1) << 1;
}

template<typename T>
T align_down_n(T x, uint32_t n)
{
    return x & ~(n-1);
}

template<typename T>
T align_up_n(T x, uint32_t n)
{
    return (x + (n-1)) & ~(n-1);
}

} // namespace npu_toolkit