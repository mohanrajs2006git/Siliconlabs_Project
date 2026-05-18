#include "tflite_micro_model_config.h"

#include <new>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cassert>


#include "ml/utils/string.hpp"
#include "profiling.hpp"




using namespace cpputils;


namespace profiling
{


constexpr const char* const LINE_DIVIDER = "----------------------";


static void clear_visited_flags(Profiler *profiler);
static void print_profiler_statistics(
    Profiler *profiler,
    logging::Logger* logger,
    int level,
    char *level_str,
    bool pretty_print,
    int precision
);
static void print_profiler_metric(
    Profiler *profiler,
    logging::Logger* logger,
    int level,
    char *level_str,
    bool pretty_print,
    int precision
);


bool create(const char* name, Profiler* &profiler, Profiler* parent)
{
    if(strstr(name, Fullname::SEPARATOR) != nullptr)
    {
        assert(!"Profiler names must not contain ::");
        return false;
    }

    Fullname fullname;

    Fullname::create(fullname, name, parent);

    const uint32_t alloc_size = Profiler::required_alloc_size(name);
    uint8_t* ptr = static_cast<uint8_t*>(malloc(alloc_size));
    if(ptr == nullptr)
    {
        return false;
    }

    char* name_ptr = (char*)(ptr + sizeof(Profiler));
    strcpy(name_ptr, name);
    profiler = new(ptr)Profiler(ptr, name_ptr);
    assert((uintptr_t)ptr == (uintptr_t)profiler);

    if(parent != nullptr)
    {
        if(!profiler->parent(parent))
        {
            free(profiler);
            return false;
        }
    }

    return true;
}

bool destroy(Profiler* profiler)
{
    for(auto child : profiler->_children)
    {
        destroy(child);
    }
    profiler->_children.clear();

    free(profiler);

    return true;
}


bool reset(Profiler* profiler, bool reset_children)
{
    if(profiler == nullptr)
    {
        return false;
    }

    profiler->reset();

    if(reset_children)
    {
        for(auto child : profiler->_children)
        {
            reset(child, true);
        }
    }

    return true;
}


void print_metrics(
    const Profiler* profiler,
    logging::Logger *logger,
    void (*additional_callback)(
        const Profiler* profiler,
        logging::Logger *logger,
        void *arg
    ),
    void* additional_callback_arg,
    bool pretty_print,
    int precision
)
{
    if(profiler == nullptr)
    {
        return;
    }

    if(precision < 0)
    {
        precision = pretty_print ? 2 : 6;
    }

    clear_visited_flags(const_cast<Profiler*>(profiler));

    char level_str[64];

    if(pretty_print)
    {
        logger->info(LINE_DIVIDER);
    }
    logger->info("Profiler Metrics:");
    if(pretty_print)
    {
        logger->info(LINE_DIVIDER);
    }
    if(additional_callback != nullptr)
    {
        additional_callback(profiler, logger, additional_callback_arg);
    }

    print_profiler_metric(
        const_cast<Profiler*>(profiler),
        logger,
        0,
        level_str,
        pretty_print,
        precision
    );

    if(pretty_print)
    {
        logger->info(LINE_DIVIDER);
    }
}


void print_stats(
    const Profiler* profiler,
    logging::Logger *logger,
    bool pretty_print,
    int precision
)
{
    if(profiler == nullptr)
    {
        return;
    }

    if(precision < 0)
    {
        precision = pretty_print ? 2 : 6;
    }

    clear_visited_flags(const_cast<Profiler*>(profiler));

    char level_str[64];

    if(pretty_print)
    {
        logger->info(LINE_DIVIDER);
    }

    print_profiler_statistics(
        const_cast<Profiler*>(profiler),
        logger,
        0,
        level_str,
        pretty_print,
        precision
    );

    if(pretty_print)
    {
        logger->info(LINE_DIVIDER);
    }
}

uint32_t calculate_total_children_cpu_cycles(const Profiler* profiler, bool is_root)
{
    uint32_t total_cycles = 0;

    if(!is_root && !profiler->flags().is_set(Flag::ExcludeFromTotalChildrenCyclesReport))
    {
        const auto& s = profiler->stats();
        total_cycles += s.cpu_cycles;
    }
    for(auto child : profiler->_children)
    {
        total_cycles += calculate_total_children_cpu_cycles(child, false);
    }

    return total_cycles;
}

uint32_t calculate_total_children_accelerator_cycles(const Profiler* profiler, bool is_root)
{
    uint32_t total_cycles = 0;

    if(!is_root && !profiler->flags().is_set(Flag::ExcludeFromTotalChildrenCyclesReport))
    {
        const auto& s = profiler->stats();
        total_cycles += s.accelerator_cycles;
    }
    for(auto child : profiler->_children)
    {
        total_cycles += calculate_total_children_accelerator_cycles(child, false);
    }

    return total_cycles;
}



/** --------------------------------------------------------------------------------------------
 *  Internal functions
 * -------------------------------------------------------------------------------------------- **/


static void clear_visited_flags(Profiler *profiler)
{
    profiler->_was_visited = false;
    for(auto child : profiler->_children)
    {
        clear_visited_flags(child);
    }
}


static void print_profiler_statistics(
    Profiler *profiler,
    logging::Logger* logger,
    int level,
    char *level_str,
    bool pretty_print,
    int precision
)
{
    const auto& stats = profiler->stats();
    const auto metrics = profiler->metrics_including_children();
    const auto& flags = profiler->flags();
    bool printed_name = false;

    if(pretty_print)
    {
        level_str[level*2] = 0;
        for(int i = level*2-1; i >= 0; --i)
        {
            level_str[i] = ' ';
        }
    }
    else
    {
        level_str[0] = 0;
    }

    const auto print_profiler_name_if_necessary = [logger, profiler, &printed_name, level_str]()
    {
        if(!printed_name)
        {
            printed_name = true;
            logger->info("%s%s", level_str, profiler->_name);
        }
    };


    if(stats.time_us > 0)
    {
        print_profiler_name_if_necessary();
        logger->info("%s  %9s", level_str, format_microseconds_to_milliseconds(stats.time_us));
    }

    if(stats.cpu_cycles != 0)
    {
        print_profiler_name_if_necessary();
        logger->info("%s %8s CPU cycles%s",
                level_str,
                format_units(stats.cpu_cycles, precision),
                flags.is_set(Flag::ReportsFreeRunningCpuCycles) ? " (free running)" :
                flags.is_set(Flag::ExcludeFromTotalChildrenCyclesReport) ? " (excluded from child sum)" : "");
    }

    uint32_t children_accelerator_cycles = 0;
    uint32_t children_cpu_cycles = 0;

    if(flags.is_set(Flag::ReportTotalChildrenCycles))
    {
        children_accelerator_cycles = calculate_total_children_accelerator_cycles(profiler);
        children_cpu_cycles = calculate_total_children_cpu_cycles(profiler);

        if(children_cpu_cycles > 0)
        {
            print_profiler_name_if_necessary();
            logger->info("%s %8s CPU cycles (sum of all child profilers)", level_str, format_units(children_cpu_cycles, precision));
        }

        if(children_accelerator_cycles > 0)
        {
            print_profiler_name_if_necessary();
            logger->info("%s %8s Accelerator cycles (sum of all child profilers)", level_str, format_units(children_accelerator_cycles, precision));
        }
    }

    if(!flags.is_set(Flag::ExcludeStatsFromReport))
    {
        if(stats.accelerator_cycles != 0)
        {
            print_profiler_name_if_necessary();
            logger->info("%s %8s Accelerator cycles", level_str, format_units(stats.accelerator_cycles, precision));
        }

        if(stats.time_us > 0)
        {
            if(metrics.ops != 0)
            {
                print_profiler_name_if_necessary();
                logger->info("%s %8s OP/s", level_str, format_rate(metrics.ops, stats.time_us, precision));
            }

            if(metrics.macs != 0)
            {
                print_profiler_name_if_necessary();
                logger->info("%s %8s MAC/s", level_str, format_rate(metrics.macs, stats.time_us, precision));

                if(children_accelerator_cycles != 0)
                {
                    logger->info("%s %8.1f MAC/cycle", level_str, (float)metrics.macs / children_accelerator_cycles);
                }
                else if(stats.accelerator_cycles != 0)
                {
                    logger->info("%s %8.1f MAC/cycle", level_str, (float)metrics.macs / stats.accelerator_cycles);
                }
                else if(children_cpu_cycles != 0)
                {
                    logger->info("%s %8.1f MAC/cycle", level_str, (float)metrics.macs / children_cpu_cycles);
                }
                else if(stats.cpu_cycles != 0)
                {
                    logger->info("%s %8.1f MAC/cycle", level_str, (float)metrics.macs / stats.cpu_cycles);
                }
            }
        }

        #ifdef PROFILER_CUSTOM_STATS_ENABLED
        for(auto& e : profiler->custom_stats)
        {
            print_profiler_name_if_necessary();
            const auto& name = std::get<0>(e);
            const auto& value = std::get<1>(e);
            logger->info("%s %s=%d", level_str, name, value);
        }
        #endif // PROFILER_CUSTOM_STATS_ENABLED
    }

    profiler->_was_visited = true;

    for(auto child : profiler->_children)
    {
        if(child->_was_visited)
        {
            continue;
        }

        print_profiler_statistics(child, logger, level+1, level_str, pretty_print, precision);
    }
}


static void print_profiler_metric(
    Profiler *profiler,
    logging::Logger *logger,
    int level,
    char *level_str,
    bool pretty_print,
    int precision
)
{
    const auto metrics = profiler->metrics_including_children();

    if(pretty_print)
    {
        level_str[level*2] = 0;
        for(int i = level*2-1; i >= 0; --i)
        {
            level_str[i] = ' ';
        }
    }
    else
    {
        level_str[0] = 0;
    }

    if(metrics.ops != 0 || metrics.macs != 0)
    {
        logger->info("%s%s", level_str, profiler->name());
        if(metrics.ops != 0)
        {
            logger->info("%s %7s OPs", level_str, format_units(metrics.ops, precision));
        }
        if(metrics.macs != 0)
        {
            logger->info("%s %7s MACs", level_str, format_units(metrics.macs, precision));
        }
    }
    profiler->_was_visited = true;

    for(auto child : profiler->_children)
    {
        if(child->_was_visited)
        {
            continue;
        }

        print_profiler_metric(child, logger, level+1, level_str, pretty_print, precision);
    }
}



} // namespace profiling
