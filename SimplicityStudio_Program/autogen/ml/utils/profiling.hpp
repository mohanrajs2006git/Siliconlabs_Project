#include "tflite_micro_model_config.h"
#pragma once

#include "ml/utils/logging.hpp"
#include "ml/utils/profiler.hpp"

namespace profiling
{

bool create(const char* name, Profiler* &profiler, Profiler* parent=nullptr);
bool destroy(Profiler* profiler);

bool reset(Profiler* profiler, bool reset_children=true);

void print_metrics(
  const Profiler* profiler,
  logging::Logger *logger,
  void (*additional_callback)(
    const Profiler* profiler,
    logging::Logger *logger,
    void *arg
  ) = nullptr,
  void* additional_callback_arg = nullptr,
  bool pretty_print = true,
  int precision = -1

);
void print_stats(
  const Profiler* profiler,
  logging::Logger *logger,
  bool pretty_print = true,
  int precision = -1
);
uint32_t calculate_total_children_cpu_cycles(const Profiler* profiler, bool is_root = true);
uint32_t calculate_total_children_accelerator_cycles(const Profiler* profiler, bool is_root = true);



} // namespace profiling
