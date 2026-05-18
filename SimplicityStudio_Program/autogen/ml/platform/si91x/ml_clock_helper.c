#include "tflite_micro_model_config.h"

#include <assert.h>
#include <stdint.h>
#include "ml/platform/ml_clock_helper.h"
#include "system_si91x.h"
#include "si91x_device.h"
#include "si91x_mvp.h"
#include "rsi_rom_clks.h"
#include "sl_si91x_clock_manager.h"
#include "ml/platform/ml_clock_helper.h"



#define MAX_CLOCK_RATE_HZ 180000000 // 180MHz
#define MVP_CLK_ENABLE_BIT 25



bool ml_configure_clocks_to_max_rate()
{
  #ifdef NPU_TOOLKIT_CPU_CLOCK
    return ml_set_cpu_clock_rate(NPU_TOOLKIT_CPU_CLOCK);
  #else
    return ml_set_cpu_clock_rate(MAX_CLOCK_RATE_HZ);
  #endif
}

bool ml_set_cpu_clock_rate(uint32_t hz)
{
  int32 status;

  sl_si91x_clock_manager_m4_set_core_clk(M4_SOCPLLCLK, hz);
  sl_si91x_clock_manager_set_pll_freq(INFT_PLL, hz, PLL_REF_CLK_VAL_XTAL);

  status = ROMAPI_M4SS_CLK_API->clk_qspi_clk_config(M4CLK, QSPI_SOCPLLCLK, 0, 0, 0);
  if (status != RSI_OK)
  {
    assert(!"Failed to set QSPI Clock as SOC PLL clock");
    return false;
  }

  status = RSI_CLK_Qspi2ClkConfig(M4CLK, QSPI_SOCPLLCLK, 0, 0, 0);
  if (status != RSI_OK)
  {
    assert(!"Failed to set QSPI 2 Clock as SOC PLL clock");
    return false;
  }

  return true;
}

uint32_t ml_get_cpu_clock_frequency()
{
  return system_clocks.soc_clock;
}


void ml_set_accelerator_clock_enabled(bool enabled)
{
  if(enabled)
  {
    M4CLK->CLK_ENABLE_SET_REG1 |= BIT(MVP_CLK_ENABLE_BIT);
    M4CLK->DYN_CLK_GATE_DISABLE_REG |= BIT(MVP_CLK_ENABLE_BIT);
  }
  else
  {
    M4CLK->CLK_ENABLE_CLEAR_REG1 = BIT(MVP_CLK_ENABLE_BIT);
    M4CLK->DYN_CLK_GATE_DISABLE_REG &= ~(BIT(MVP_CLK_ENABLE_BIT));
  }
}


