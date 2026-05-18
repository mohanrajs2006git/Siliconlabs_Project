#include "speaker.h"
#include "sl_si91x_usart.h"
#include "sl_sleeptimer.h"
#include <string.h>
#include <stdio.h>
#include "rsi_debug.h"
#define USART_BAUDRATE        9600
#define MS_DELAY_COUNTER      4600
#define ULP_BANK_OFFSET       0x800
#define TX_BUF_MEMORY         (ULP_SRAM_START_ADDR + (1 * ULP_BANK_OFFSET))

#define DFMINI_START_BYTE     0x7E
#define DFMINI_VERSION        0xFF
#define DFMINI_CMD_LEN        0x06
#define DFMINI_FEEDBACK_OFF   0x00
#define DFMINI_END_BYTE       0xEF
#define DFMINI_PACKET_SIZE    10

#define DFMINI_CMD_PLAY_TRACK 0x03
#define DFMINI_CMD_SET_VOLUME 0x06
#define DFMINI_CMD_STOP       0x16

#define DFMINI_VOLUME_LEVEL   25
#define DFMINI_BOOT_DELAY_MS  1500
#define DFMINI_CMD_GAP_MS     300
#define DFMINI_TX_TIMEOUT_MS  500

static sl_usart_handle_t  speaker_usart_handle;
static uint8_t            dfmini_cmd_buf[DFMINI_PACKET_SIZE];
static volatile boolean_t tx_complete     = false;
static boolean_t          vol_initialized = false;

static void speaker_delay(uint32_t ms)
{
  for (uint32_t x = 0; x < MS_DELAY_COUNTER * ms; x++) {
    __NOP();
  }
}

static void dfmini_build_packet(uint8_t *buf, uint8_t cmd,
                                uint8_t param1, uint8_t param2)
{
  buf[0] = DFMINI_START_BYTE;
  buf[1] = DFMINI_VERSION;
  buf[2] = DFMINI_CMD_LEN;
  buf[3] = cmd;
  buf[4] = DFMINI_FEEDBACK_OFF;
  buf[5] = param1;
  buf[6] = param2;

  int16_t checksum = 0;
  for (uint8_t i = 1; i <= 6; i++) {
    checksum -= (int16_t)buf[i];
  }
  buf[7] = (uint8_t)((checksum >> 8) & 0xFF);
  buf[8] = (uint8_t)(checksum & 0xFF);
  buf[9] = DFMINI_END_BYTE;
}

static sl_status_t dfmini_send(uint8_t cmd, uint8_t param1, uint8_t param2)
{
  tx_complete = false;

  dfmini_build_packet(dfmini_cmd_buf, cmd, param1, param2);
  memcpy((uint8_t *)TX_BUF_MEMORY, dfmini_cmd_buf, DFMINI_PACKET_SIZE);

  sl_status_t status = sl_si91x_usart_send_data(speaker_usart_handle,
                                                 (uint8_t *)TX_BUF_MEMORY,
                                                 DFMINI_PACKET_SIZE);
  if (status != SL_STATUS_OK) {
    DEBUGOUT("[SPEAKER] UART send failed: 0x%lx\r\n", (unsigned long)status);
    return status;
  }

  uint32_t start = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
  while (!tx_complete) {
    uint32_t now = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
    if ((now - start) >= DFMINI_TX_TIMEOUT_MS) {
      DEBUGOUT("[SPEAKER] TX timeout cmd=0x%02X\r\n", cmd);
      break;
    }
  }
  tx_complete = false;
  speaker_delay(DFMINI_CMD_GAP_MS);

  return SL_STATUS_OK;
}

void speaker_uart_callback_event(uint32_t event)
{
  if (event == SL_USART_EVENT_SEND_COMPLETE) {
    tx_complete = true;
  }
}

void speaker_init(void)
{
  sl_status_t status;
  sl_si91x_usart_control_config_t usart_config;
  sl_si91x_usart_control_config_t get_config;

  usart_config.baudrate      = USART_BAUDRATE;
  usart_config.mode          = SL_USART_MODE_ASYNCHRONOUS;
  usart_config.parity        = SL_USART_NO_PARITY;
  usart_config.stopbits      = SL_USART_STOP_BITS_1;
  usart_config.hwflowcontrol = SL_USART_FLOW_CONTROL_NONE;
  usart_config.databits      = SL_USART_DATA_BITS_8;
  usart_config.misc_control  = SL_USART_MISC_CONTROL_NONE;
  usart_config.usart_module  = ULPUART;
  usart_config.config_enable = ENABLE;
  usart_config.synch_mode    = DISABLE;

  do {
    status = sl_si91x_usart_init(ULPUART, &speaker_usart_handle);
    if (status != SL_STATUS_OK) {
      DEBUGOUT("[SPEAKER] UART init failed: 0x%lx\r\n", (unsigned long)status);
      break;
    }

    status = sl_si91x_usart_set_configuration(speaker_usart_handle,
                                               &usart_config);
    if (status != SL_STATUS_OK) {
      DEBUGOUT("[SPEAKER] UART config failed: 0x%lx\r\n", (unsigned long)status);
      break;
    }

    status = sl_si91x_usart_multiple_instance_register_event_callback(
               ULPUART, speaker_uart_callback_event);
    if (status != SL_STATUS_OK) {
      DEBUGOUT("[SPEAKER] Callback register failed: 0x%lx\r\n",
             (unsigned long)status);
      break;
    }

    sl_si91x_usart_get_configurations(ULPUART, &get_config);
    DEBUGOUT("[SPEAKER] UART ready. Baud=%ld\r\n", get_config.baudrate);

    DEBUGOUT("[SPEAKER] Waiting %dms for DFMini boot...\r\n",
           DFMINI_BOOT_DELAY_MS);
    speaker_delay(DFMINI_BOOT_DELAY_MS);
    DEBUGOUT("[SPEAKER] DFMini booted.\r\n");

    dfmini_send(DFMINI_CMD_SET_VOLUME, 0x00, DFMINI_VOLUME_LEVEL);
    vol_initialized = true;
    DEBUGOUT("[SPEAKER] Volume=%d/30. Speaker ready.\r\n", DFMINI_VOLUME_LEVEL);

  } while (false);
}

void speaker_play(uint8_t track)
{
  if (!vol_initialized) {
    DEBUGOUT("[SPEAKER] ERROR: Not initialized!\r\n");
    return;
  }

  DEBUGOUT("[SPEAKER] Playing track %d\r\n", track);

  dfmini_send(DFMINI_CMD_STOP, 0x00, 0x00);
  speaker_delay(DFMINI_CMD_GAP_MS);
  sl_status_t s = dfmini_send(DFMINI_CMD_PLAY_TRACK, 0x00, track);

  if (s == SL_STATUS_OK) {
    DEBUGOUT("[SPEAKER] Track %d sent OK\r\n", track);
  } else {
    DEBUGOUT("[SPEAKER] Track %d FAILED 0x%lx\r\n", track, (unsigned long)s);
  }
}
