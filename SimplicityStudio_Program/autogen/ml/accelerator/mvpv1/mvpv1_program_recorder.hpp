#include "tflite_micro_model_config.h"
#pragma once
#ifdef TFLITE_MICRO_RECORDER_ENABLED

#include "ml/third_party/tflm/tlmk-kernel_util.h"
#include "ml/accelerator/mvpv1/mvpv1_accelerator_utils.hpp"

#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
#include "mvp/driver.hpp"
#endif


namespace npu_toolkit
{

enum class ProgramType : uint8_t
{
  General = 0,
  FusedRelu = 1,
  Conv2dInputScaler = 2,
};


/**
 * Flag that the given ALU register is constant and does not change throughout program execution.
 * This is recorded by the NPU Python wrapper
 * which is used by the MVP model compiler when generating program "deltas".
*/
uint8_t mvpv1_kernel_flag_alu_register_as_const(uint8_t reg_id);

/**
 * Flag that the given ALU register is initialized before program execution.
 * This is recorded by the NPU Python wrapper
 * which is used by the MVP model compiler when generating program "deltas".
*/
uint8_t mvpv1_kernel_flag_alu_register_as_init(uint8_t reg_id);

void mvpv1_kernel_record_program_run_cycles(int cycles);
void mvpv1_kernel_record_program_type(ProgramType type);
void mvpv1_kernel_record_mvp_array_base_address(int index, const void* addr);
int mvpv1_kernel_record_mvp_loop_as_used(int index);
int mvpv1_kernel_record_mvp_array_as_used(int index);
int mvpv1_kernel_record_mvp_instr_as_used(int index);
void mvpv1_kernel_record_mvp_program_callback(const void *program);
void mvpv1_kernel_record_layer_tensor(const char* name, const TfLiteEvalTensor *tensor);
void mvpv1_kernel_record_memory_stats();
void mvpv1_kernel_record_reset();
TfLiteStatus mvpv1_kernel_record_layer_callback(
  TfliteMicroSection section,
  bool is_start,
  int index,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus status,
  void* arg
);

#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
void mvpv1_kernel_record_program_usage(const ::mvpv1::ProgramBuilder& builder);
#endif


} // namespace npu_toolkit

#else

#define mvpv1_kernel_record_program_run_cycles(...)
#define mvpv1_kernel_record_program_type(...)
#define mvpv1_kernel_record_mvp_array_base_address(...)
#define mvpv1_kernel_record_mvp_program_callback(...)
#define mvpv1_kernel_flag_alu_register_as_const(reg_id) reg_id
#define mvpv1_kernel_flag_alu_register_as_init(reg_id) reg_id
#define mvpv1_kernel_record_mvp_array_as_used(i) i
#define mvpv1_kernel_record_mvp_loop_as_used(i) i
#define mvpv1_kernel_record_mvp_instr_as_used(i) i
#define mvpv1_kernel_record_program_usage(x)
#define mvpv1_kernel_record_layer_tensor(...) _mvpv1_kernel_record_placeholder()
#define mvpv1_kernel_record_memory_stats() _mvpv1_kernel_record_placeholder()
#define mvpv1_kernel_record_reset() _mvpv1_kernel_record_placeholder()


static inline void _mvpv1_kernel_record_placeholder(){}

#endif // TFLITE_MICRO_RECORDER_ENABLED



#define MVPV1_SET_ARRAY_BASE_ADDRESS(index, address) \
  mvpv1_kernel_record_mvp_array_base_address(index, (const void*)address);




