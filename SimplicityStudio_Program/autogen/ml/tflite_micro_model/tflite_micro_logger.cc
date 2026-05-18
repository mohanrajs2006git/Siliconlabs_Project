#include "tflite_micro_model_config.h"

#include <new>
#include <cstdlib>
#include <cassert>

#include "ml/tflite_micro_model/tflite_micro_logger.hpp"



namespace npu_toolkit
{

Logger* DLL_EXPORT(TfliteMicroLogger_get_logger(LoggerId id));
void DLL_EXPORT(TfliteMicroLogger_set_logger(LoggerId id, Logger* logger));


Logger& get_logger(
    LoggerId id,
    LogLevel default_level
)
{
    if(TfliteMicroLogger_get_logger(id) == nullptr)
    {
        void *instance_buffer = malloc(sizeof(Logger));
        assert(instance_buffer != nullptr);
        TfliteMicroLogger_set_logger(id, new(instance_buffer)Logger(
            to_str(id),
            default_level,
            logging::Flag::Newline
        ));
    }
    return *TfliteMicroLogger_get_logger(id);
}


#ifndef NPU_TOOLKIT_DLL_IMPORT
static Logger* TfliteMicroLogger_loggers[(int)LoggerId::Count];

Logger* TfliteMicroLogger_get_logger(LoggerId id)
{
    return TfliteMicroLogger_loggers[(int)id];
}

void TfliteMicroLogger_set_logger(LoggerId id, Logger* logger)
{
    TfliteMicroLogger_loggers[(int)id] = logger;
}

#endif // NPU_TOOLKIT_DLL_IMPORT


} // namespace npu_toolkit
