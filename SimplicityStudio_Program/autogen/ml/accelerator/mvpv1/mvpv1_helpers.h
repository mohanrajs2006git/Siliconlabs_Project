#include "tflite_micro_model_config.h"
#pragma once

#include "em_device.h"


#ifdef __cplusplus
extern "C" {
#endif


void mvpv1_dump_registers(const void *program_context);


void mvpv1_dump_instance_program_registers(int program_index);
void mvpv1_dump_program_registers(const void *program_context, int program_index);


void mvpv1_dump_alu_registers(const MVP_ALU_TypeDef* alu_regs);
void mvpv1_dump_alu_register(unsigned index, const MVP_ALU_TypeDef* alu_reg);

void mvpv1_dump_array_registers(const MVP_ARRAY_TypeDef* array_regs);
void mvpv1_dump_array_register(unsigned index, const MVP_ARRAY_TypeDef* array_reg);

void mvpv1_dump_loop_registers(const MVP_LOOP_TypeDef* loop_regs);
void mvpv1_dump_loop_register(unsigned index, const MVP_LOOP_TypeDef* loop_regs);

void mvpv1_dump_instr_registers(const MVP_INSTR_TypeDef* instr_regs);
void mvpv1_dump_instr_register(unsigned index, const MVP_INSTR_TypeDef* instr_regs);

void mvpv1_irq_semaphore_wait();
void mvpv1_irq_semaphore_signal();

void mvpv1_set_dump_writer(void (*writer)(const char*,va_list,void*), void *arg);


#ifdef __cplusplus
}
#endif
