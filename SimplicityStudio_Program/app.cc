#include "app.h"
#include "audio_classifier.h"
#include "ml/platform/ml_clock_helper.h"
#include "sl_sleeptimer.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "led_helper.hpp"
#include "button.h"
#include "speaker.h"
#include "rsi_debug.h"
#ifdef __cplusplus
}
#endif

#include "mic_helper.hpp"

extern void app_init(void)
{
  DEBUGOUT("\r\n\r\n");
  DEBUGOUT(" ==========================================\r\n");
  DEBUGOUT(" |   PATIENT MONITORING SYSTEM  v1.0      |\r\n");
  DEBUGOUT(" |   SiWx917 + ML + Buttons + Speaker     |\r\n");
  DEBUGOUT(" ==========================================\r\n");
  DEBUGOUT("\r\n");

  DEBUGOUT(" [BOOT] Configuring clocks...\r\n");
  ml_configure_clocks_to_max_rate();
  DEBUGOUT(" [BOOT] Clocks     : OK\r\n");

  DEBUGOUT(" [BOOT] Init LEDs...\r\n");
  led_init();
  DEBUGOUT(" [BOOT] LEDs       : OK\r\n");

  // speaker_init() is NOT here — it is called inside
  // audio_classifier_task() AFTER the ML model loads.
  // Calling it here causes ULP UART to hang before any output.

  DEBUGOUT(" [BOOT] Enable Microphone...\r\n");
  enable_microphone();
  DEBUGOUT(" [BOOT] Microphone : OK\r\n");

  DEBUGOUT(" [BOOT] Init Buttons...\r\n");
  button_init();
  DEBUGOUT(" [BOOT] Buttons    : OK\r\n");

  DEBUGOUT("\r\n");
  DEBUGOUT(" [BOOT] All done. Starting ML...\r\n");
  DEBUGOUT("\r\n");
}

extern void app_process_action(void)
{
  audio_classifier_task();
}
