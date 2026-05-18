#include "tflite_micro_model_config.h"

#include <stdint.h>
#include "em_device.h"

#if __has_include("sl_sleeptimer_config.h")
#include "sl_sleeptimer_config.h"

extern uint32_t sleeptimer_hal_get_timer_frequency(void);

#if SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_RTCC
    #define MICROSECOND_TIMER_CNT RTCC->CNT

#elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_BURTC
    #define MICROSECOND_TIMER_CNT BURTC->CNT

#elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_RTC
    #define MICROSECOND_TIMER_CNT RTC->CNT

#elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_SYSRTC
    #define MICROSECOND_TIMER_CNT SYSRTC0->CNT

#elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_DEFAULT
    #define MICROSECOND_TIMER_CNT SYSRTC0->CNT

#else
    #error ARM platform not supported
#endif


uint32_t microsecond_timer_get_timestamp()
{
    static uint32_t timer_scaler = 0;
    if(timer_scaler == 0)
    {
        const uint32_t timer_freq = sleeptimer_hal_get_timer_frequency();
        timer_scaler = 1000000 / timer_freq;
    }

    return MICROSECOND_TIMER_CNT * timer_scaler;
}

#endif //  __has_include("sl_sleeptimer_config.h")