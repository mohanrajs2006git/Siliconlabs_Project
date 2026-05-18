#include "button.h"
#include "speaker.h"
#include "sl_si91x_driver_gpio.h"
#include "sl_sleeptimer.h"
#include <stdbool.h>
#include <stdio.h>
#include "rsi_debug.h"
#define BTN_PORT  0

static const uint8_t btn_pin[4] = {10,  6, 12,  7};
static const uint8_t btn_pad[4] = { 5,  1,  7,  2};

static const char *btn_label[4] = {
  "I NEED FOOD",
  "I NEED WATER",
  "I NEED TO GO RESTROOM",
  "EMERGENCY"
};

static const char *btn_source[4] = {
  "BTN0 | GPIO10 | Food    ",
  "BTN1 | GPIO6  | Water   ",
  "BTN2 | GPIO12 | Restroom",
  "BTN3 | GPIO7  | Emergency"
};

static const uint8_t btn_track[4] = {
  TRACK_FOOD,
  TRACK_WATER,
  TRACK_RESTROOM,
  TRACK_EMERGENCY
};

static bool     button_state[4]     = {false};
static uint32_t button_count[4]     = {0};
static uint32_t last_poll_ms        = 0;

#define POLL_INTERVAL_MS    50
#define ACTIVE_DURATION_MS  4000

volatile bool   button_active       = false;
static uint32_t button_active_until = 0;
static volatile uint8_t pending_track = 0;

static void configure_input_pin(uint8_t pin, uint8_t pad)
{
  sl_si91x_gpio_driver_enable_clock(M4CLK_GPIO);
  sl_si91x_gpio_driver_enable_pad_selection(pad);
  sl_si91x_gpio_driver_enable_pad_receiver(pin);
  sl_si91x_gpio_driver_set_pin_direction(BTN_PORT, pin, GPIO_INPUT);
  sl_si91x_gpio_driver_select_pad_driver_disable_state(pin, GPIO_PULLUP);
}

static void print_button_alert(int i, uint32_t count)
{
  if (i == 3) {
    DEBUGOUT("\r\n");
    DEBUGOUT("  /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\\r\n");
    DEBUGOUT(" +------------------------------------------+\r\n");
    DEBUGOUT(" |        *** EMERGENCY ALERT ***           |\r\n");
    DEBUGOUT(" +------------------------------------------+\r\n");
    DEBUGOUT(" |  Source  : %-30s|\r\n", btn_source[i]);
    DEBUGOUT(" |  Request : %-30s|\r\n", btn_label[i]);
    DEBUGOUT(" |  Track   : %d                             |\r\n", btn_track[i]);
    DEBUGOUT(" |  Count   : %-4lu                          |\r\n", (unsigned long)count);
    DEBUGOUT(" +------------------------------------------+\r\n");
    DEBUGOUT("  /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\ /!\\\r\n");
    DEBUGOUT("\r\n");
  } else {
    DEBUGOUT("\r\n");
    DEBUGOUT(" +------------------------------------------+\r\n");
    DEBUGOUT(" |         PATIENT REQUEST  [BUTTON]        |\r\n");
    DEBUGOUT(" +------------------------------------------+\r\n");
    DEBUGOUT(" |  Source  : %-30s|\r\n", btn_source[i]);
    DEBUGOUT(" |  Request : %-30s|\r\n", btn_label[i]);
    DEBUGOUT(" |  Track   : %d                             |\r\n", btn_track[i]);
    DEBUGOUT(" |  Count   : %-4lu                          |\r\n", (unsigned long)count);
    DEBUGOUT(" +------------------------------------------+\r\n");
    DEBUGOUT("\r\n");
  }
}

void button_init(void)
{
  sl_gpio_driver_init();

  for (int i = 0; i < 4; i++) {
    configure_input_pin(btn_pin[i], btn_pad[i]);
  }

  DEBUGOUT(" +------------------------------------------+\r\n");
  DEBUGOUT(" |      BUTTON SYSTEM INITIALIZED           |\r\n");
  DEBUGOUT(" +------------------------------------------+\r\n");
  DEBUGOUT(" |  BTN0 GPIO10  ->  I Need Food  (Trk 5)  |\r\n");
  DEBUGOUT(" |  BTN1 GPIO6   ->  I Need Water (Trk 6)  |\r\n");
  DEBUGOUT(" |  BTN2 GPIO12  ->  Restroom     (Trk 1)  |\r\n");
  DEBUGOUT(" |  BTN3 GPIO7   ->  Emergency    (Trk 3)  |\r\n");
  DEBUGOUT(" +------------------------------------------+\r\n");

  DEBUGOUT(" |  Pin States  : ");
  for (int i = 0; i < 4; i++) {
    sl_gpio_t g = { (sl_gpio_port_t)BTN_PORT, btn_pin[i] };
    uint8_t v = 0xFF;
    sl_gpio_driver_get_pin(&g, &v);
    DEBUGOUT("G%d=%d ", btn_pin[i], v);
  }
  DEBUGOUT("          |\r\n");
  DEBUGOUT(" +------------------------------------------+\r\n");
  DEBUGOUT("\r\n");
}

void button_task(void)
{
  uint32_t now = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

  // Play pending track — isolated from GPIO read
  if (pending_track > 0) {
    uint8_t t  = pending_track;
    pending_track = 0;
    speaker_play(t);
    return;
  }

  // Clear button_active after timeout
  if (button_active && now >= button_active_until) {
    button_active = false;
    DEBUGOUT(" [SYSTEM] ML Detection Resumed\r\n\r\n");
  }

  // Throttle
  if ((now - last_poll_ms) < POLL_INTERVAL_MS) return;
  last_poll_ms = now;

  for (int i = 0; i < 4; i++) {
    sl_gpio_t g    = { (sl_gpio_port_t)BTN_PORT, btn_pin[i] };
    uint8_t   val  = 1;

    sl_status_t st = sl_gpio_driver_get_pin(&g, &val);
    if (st != SL_STATUS_OK) {
      DEBUGOUT(" [BTN] ERROR GPIO%d st=0x%lx\r\n",
             btn_pin[i], (unsigned long)st);
      continue;
    }

    bool pressed = (val == 0);

    if (pressed && !button_state[i]) {
      button_count[i]++;
      button_active       = true;
      button_active_until = now + ACTIVE_DURATION_MS;
      print_button_alert(i, button_count[i]);
      pending_track = btn_track[i];
    }

    if (!pressed && button_state[i]) {
      DEBUGOUT(" [BTN] %s Released\r\n\r\n", btn_label[i]);
    }

    button_state[i] = pressed;
  }
}
