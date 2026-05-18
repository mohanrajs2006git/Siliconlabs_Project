#include "tflite_micro_model_config.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <algorithm>

#include "ml/accelerator/mvpv1/mvpv1_helpers.h"


constexpr const unsigned NUM_DIMS = 3;

#define ITEM(name) (((MVP_TypeDef*)0)->name)
#define ITEM_COUNT(name) ((unsigned)(sizeof(ITEM(name)) / sizeof(ITEM(name)[0])))
#define PRINTF(fmt, ...) _printf(fmt, ## __VA_ARGS__)

typedef struct
{
    MVP_ALU_TypeDef ALU[ITEM_COUNT(ALU)];
    MVP_ARRAY_TypeDef ARRAY[ITEM_COUNT(ARRAY)];
    MVP_LOOP_TypeDef LOOP[ITEM_COUNT(LOOP)];
    MVP_INSTR_TypeDef INSTR[ITEM_COUNT(INSTR)];
} MVP_ProgramContext;



/*************************************************************************************************/
template<typename T>
bool has_values(const T* reg)
{
    const uint8_t* p = (uint8_t*)reg;
    for(unsigned i = 0; i < sizeof(T); ++i, ++p)
    {
        if(*p != 0)
        {
            return true;
        }
    }
    return false;
}




extern "C" {

static void _printf(const char* fmt, ...);


/*************************************************************************************************/
void mvpv1_dump_registers(const void *program_context)
{
    PRINTF("------------------------------------");

    if(MVP->EN & MVP_EN_EN)
    {
        mvpv1_dump_program_registers(program_context ? program_context : &MVP->ALU, -1);
    }
}

/*************************************************************************************************/
void mvpv1_dump_instance_program_registers(int program_index)
{
    mvpv1_dump_program_registers(&MVP->ALU, program_index);
}

/*************************************************************************************************/
void mvpv1_dump_program_registers(const void *program_context, int program_index)
{
    if(program_index != -1)
    {
        PRINTF("------------------------------------");
        PRINTF("Program %d", program_index);
    }

    const MVP_ProgramContext* prog = (const MVP_ProgramContext*)program_context;

    mvpv1_dump_alu_registers(prog->ALU);
    mvpv1_dump_array_registers(prog->ARRAY);
    mvpv1_dump_loop_registers(prog->LOOP);
    mvpv1_dump_instr_registers(prog->INSTR);
}

/*************************************************************************************************/
void mvpv1_dump_array_registers(const MVP_ARRAY_TypeDef* array_regs)
{
    for(unsigned i = 0; i < ITEM_COUNT(ARRAY); ++i)
    {
        mvpv1_dump_array_register(i, &array_regs[i]);
    }
}

/*************************************************************************************************/
void mvpv1_dump_array_register(unsigned index, const MVP_ARRAY_TypeDef* array_reg)
{
    if(!has_values(array_reg))
    {
        return;
    }
    char buffer[128];
    char *s = buffer;

    s += sprintf(s, "ARRAY%d: ADDRCFG:%08X ", index, (unsigned int)array_reg->ADDRCFG);

    const uint32_t* dims = (const uint32_t*)&array_reg->DIM0CFG;
    for (unsigned i=0; i < NUM_DIMS; ++i) {
        s += sprintf(s, "DIM%dCFG:%08X", i, (unsigned int)dims[i]);
        if(i < NUM_DIMS-1)
        {
            s += sprintf(s, " ");
        }
    }
    PRINTF(buffer);
}

/*************************************************************************************************/
void mvpv1_dump_alu_registers(const MVP_ALU_TypeDef* alu_regs)
{
    for(unsigned i = 0; i < ITEM_COUNT(ALU); ++i)
    {
        mvpv1_dump_alu_register(i, &alu_regs[i]);
    }
}

/*************************************************************************************************/
void mvpv1_dump_alu_register(unsigned index, const MVP_ALU_TypeDef* alu_reg)
{
    char buffer[128];
    sprintf(buffer, "  ALU%d: %08X", index, (unsigned int)alu_reg->REGSTATE);
    PRINTF(buffer);
}

/*************************************************************************************************/
void mvpv1_dump_loop_registers(const MVP_LOOP_TypeDef* loop_regs)
{
    for(unsigned i = 0; i < ITEM_COUNT(LOOP); ++i)
    {
        mvpv1_dump_loop_register(i, &loop_regs[i]);
    }
}

/*************************************************************************************************/
void mvpv1_dump_loop_register(unsigned index, const MVP_LOOP_TypeDef* loop_regs)
{
    if(!has_values(loop_regs))
    {
        return;
    }

    PRINTF(" LOOP%d: CFG:%08X RST:%08X",
        index,
        (unsigned int)loop_regs->CFG,
        (unsigned int)loop_regs->RST
    );
}

/*************************************************************************************************/
void mvpv1_dump_instr_registers(const MVP_INSTR_TypeDef* instr_regs)
{
    for(unsigned i = 0; i < ITEM_COUNT(INSTR); ++i)
    {
        mvpv1_dump_instr_register(i, &instr_regs[i]);
    }
}

/*************************************************************************************************/
void mvpv1_dump_instr_register(unsigned index, const MVP_INSTR_TypeDef* instr_regs)
{
    if(!has_values(instr_regs))
    {
        return;
    }

    PRINTF("INSTR%d: CFG0:%08X CFG1:%08X CFG2:%08X",
        index, (unsigned int)instr_regs->CFG0,
        (unsigned int)instr_regs->CFG1,
        (unsigned int)instr_regs->CFG2
    );
}


#ifdef __arm__

static volatile int irq_counter = 0;
void mvpv1_irq_semaphore_wait()
{
    while(irq_counter <= 0)
    {
        __WFE();
    }
    __disable_irq();
    --irq_counter;
    __enable_irq();
}

void mvpv1_irq_semaphore_signal()
{
    __disable_irq();
    ++irq_counter;
    __enable_irq();
    __SEV();
}

#else // ifdef __arm__

static volatile void* irq_sem_handler = NULL;
static void create_semaphore()
{
    if(irq_sem_handler == NULL)
    {
        irq_sem_handler = slsc_mvpv1_simulator_create_semaphore(0);
    }
}

void mvpv1_irq_semaphore_wait()
{
    create_semaphore();
    slsc_mvpv1_simulator_acquire_semaphore((void*)irq_sem_handler);
}

void mvpv1_irq_semaphore_signal()
{
    create_semaphore();
    slsc_mvpv1_simulator_release_semaphore((void*)irq_sem_handler);
}


#endif // ifdef __arm__




static void (*_dump_writer)(const char*,va_list, void*arg) = nullptr;
static void* _dump_writer_arg = nullptr;

static void _printf(const char* fmt, ...)
{
    if(_dump_writer == nullptr)
    {
        return;
    }

    va_list args;
    va_start(args, fmt);
    _dump_writer(fmt, args, _dump_writer_arg);
    va_end(args);
}

void mvpv1_set_dump_writer(void (*writer)(const char*,va_list, void*), void *arg)
{
    _dump_writer = writer;
    _dump_writer_arg = arg;
}


} // extern "C"