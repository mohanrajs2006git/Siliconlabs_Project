#include "tflite_micro_model_config.h"
#pragma once


#include <cstdint>


#include "em_device.h"
#include "ml/utils/flags_helper.hpp"
#include "ml/utils/typed_list.hpp"
#include "ml/utils/logging.hpp"

#include "ml/platform/microsecond_timer.h"
#include "ml/utils/profiler_fullname.hpp"


// By default, custom stats are only enabled for non-embedded builds
#if !defined(PROFILER_CUSTOM_STATS_ENABLED) && !defined(__arm__)
#define PROFILER_CUSTOM_STATS_ENABLED
#endif

#ifdef PROFILER_CUSTOM_STATS_ENABLED
#include "ml/utils/vector_dict.hpp"
#endif



namespace profiling
{

struct Metrics
{
    uint32_t ops = 0;
    uint32_t macs = 0;

    void operator += (const Metrics &other )
    {
        ops += other.ops;
        macs += other.macs;
    }
};


struct Stats
{
    uint32_t time_us = 0;
    uint32_t cpu_cycles = 0;
    uint32_t accelerator_cycles = 0;

    void reset()
    {
        time_us = 0;
        cpu_cycles = 0;
        accelerator_cycles = 0;
    }


    void operator += (const Stats &other )
    {
        time_us += other.time_us;
        cpu_cycles += other.cpu_cycles;
        accelerator_cycles = other.accelerator_cycles;
    }
};

struct StatsAccumulator
{
    volatile uint32_t start_base = 0;
    volatile uint32_t start_marker = 0;
    uint32_t accumulator = 0;

    void reset()
    {
        start_base = 0;
        start_marker = 0;
        accumulator = 0;
    }
};

enum class Flag  : uint16_t
{
    None = 0,
    ReportTotalChildrenCycles               = (1 << 0),
    ExcludeFromTotalChildrenCyclesReport    = (1 << 1),
    ReportsFreeRunningCpuCycles             = (1 << 2),
    ExcludeStatsFromReport                  = (1 << 3),
    TimeMeasuredBetweenStartAndStop         = (1 << 4),
};
DEFINE_ENUM_CLASS_BITMASK_OPERATORS(Flag, uint16_t)

typedef cpputils::FlagsHelper<Flag> Flags;

class Profiler;


static inline void start_cpu_cycle_counter()
{
    #ifdef __arm__
    // Enable cycle counter
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    #endif
}


static inline uint32_t get_cpu_cycles()
{
    #ifdef __arm__
    auto DWT_CYCCNT_REG ((const volatile uint32_t*)0xE0001004UL); // DWT->CYCCNT
    auto CORE_DEBUG_DEMCR = ((volatile uint32_t*)0xE000EDFC); // CoreDebug->DEMCR
    *CORE_DEBUG_DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    return *DWT_CYCCNT_REG;
    #else
    return 0; // CPU cycles on Windows/Linux don't make sense, so just return 0
    #endif
}




class Profiler
{
public:
    const char* name() const;
    const char* fullname(Fullname& fullname) const;
    void reset(void);
    const Stats& stats() const { return _stats; };
    Stats& stats() { return _stats; };
    const Metrics& metrics() const { return _metrics; };
    Metrics& metrics() { return _metrics; };
    Metrics metrics_including_children() const;
    void metrics(const Metrics& metrics);

    void flags(Flags flags);
    const Flags& flags() const { return _flags; };
    Flags& flags() { return _flags; };

    bool parent(Profiler *parent);
    constexpr Profiler* parent() const { return _parent; };
    constexpr const cpputils::TypedList<Profiler*>& children() const { return _children; }

    #ifdef PROFILER_CUSTOM_STATS_ENABLED
    cpputils::VectorDict<int32_t> custom_stats;
    #endif //  PROFILER_CUSTOM_STATS_ENABLED
    int32_t increment_custom_stat(const char* name, int32_t amount=1);
    int32_t set_custom_stat(const char* name, int32_t amount);
    int32_t get_custom_stat(const char* name) const;


    void inline start(void)
    {
        if(_state != State::Started)
        {
            start_cpu_cycle_counter();
            const uint32_t current_cycle = get_cpu_cycles();
            const uint32_t current_time_us = microsecond_timer_get_timestamp();
            if(_state == State::Stopped)
            {
                _cpu_accumulator.start_base = current_cycle;
                _time_accumulator.start_base = current_time_us;
            }
            _state = State::Started;
            _cpu_accumulator.start_marker = current_cycle;
            _time_accumulator.start_marker = current_time_us;
        }
    }

    void inline stop(void)
    {
        const volatile uint32_t stop_cpu_cycle = get_cpu_cycles();
        const volatile uint32_t stop_time_us = microsecond_timer_get_timestamp();
        update_stats(true, stop_cpu_cycle, stop_time_us);
    }

    void inline pause(void)
    {
        const volatile uint32_t stop_cpu_cycle = get_cpu_cycles();
        const volatile uint32_t stop_time_us = microsecond_timer_get_timestamp();
        update_stats(false, stop_cpu_cycle, stop_time_us);
    }


// protected:
    enum class State : uint8_t
    {
        Stopped,
        Started,
        Paused
    } _state = State::Stopped;


    Metrics _metrics;
    Stats _stats;
    cpputils::TypedList<Profiler*> _children;

    StatsAccumulator _cpu_accumulator;
    StatsAccumulator _time_accumulator;
    Flags _flags;
    Profiler *_parent = nullptr;
    const char* _name = nullptr;
    bool _was_visited = false;
    void* _object_buffer = nullptr;


    Profiler(void* object_buffer, const char* name);

    static uint32_t required_alloc_size(const char* name)
    {
        return sizeof(Profiler) + strlen(name) + 1;
    }

    void update_stats(bool stop, uint32_t stop_cpu_cycle, uint32_t stop_time_us);
    void get_child_metrics(const Profiler *profiler, Metrics &metrics) const;

    MAKE_CLASS_NON_ASSIGNABLE(Profiler);
};


} // namespace profiling

