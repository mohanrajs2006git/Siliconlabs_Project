#include "audio_classifier.h"

#include <new>
#include <cstring>
#include "recognize_commands.h"
#include "sl_ml_model_keyword_spotting.h"
#include "tflite_micro_model.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "button.h"
#include "speaker.h"
#include "rsi_debug.h"
#ifdef __cplusplus
}
#endif

using namespace npu_toolkit;

static RecognizeCommands* command_recognizer = nullptr;
static uint8_t command_recognizer_instance_buffer[sizeof(RecognizeCommands)]
               alignas(alignof(RecognizeCommands));
static int32_t detected_timeout          = 0;
static int32_t activity_timestamp        = 0;
static int32_t activity_toggle_timestamp = 0;
static uint8_t previous_score            = 0;
static int32_t previous_score_timestamp  = 0;
static int     previous_result           = 0;
int            category_count            = 0;

// [WAKE WORD] State machine -----------------------------------------------
#define WAKE_WORD_LISTEN_WINDOW_MS  5000   // ms to wait for a command after wake

typedef enum {
  WAKE_STATE_IDLE = 0,   // waiting for "assist"
  WAKE_STATE_ACTIVE,     // "assist" detected – listening for a command
} WakeWordState;

static WakeWordState wake_state              = WAKE_STATE_IDLE;
static int32_t       wake_window_expire_ms   = 0;  // absolute timestamp when window closes
// [WAKE WORD] ---------------------------------------------------------------

static uint8_t get_track_for_label(const char *label)
{
  if (label == nullptr)               return 0;
  if (strstr(label, "food"))          return TRACK_FOOD;
  if (strstr(label, "water"))         return TRACK_WATER;
  if (strstr(label, "restroom"))      return TRACK_RESTROOM;
  if (strstr(label, "assist"))        return TRACK_HELP;
  if (strstr(label, "pain"))          return TRACK_PAIN;
  if (strstr(label, "emergency"))     return TRACK_EMERGENCY;
  return 0;
}

// [WAKE WORD] Returns true if label is the wake word itself -----------------
static bool is_wake_word(const char *label)
{
  return (label != nullptr) && (strstr(label, "assist") != nullptr);
}
// [WAKE WORD] ---------------------------------------------------------------

void audio_classifier_task()
{
  uint32_t prev_loop_timestamp     = 0;
  static bool speaker_initialized  = false;

  DEBUGOUT("\r\n");
  DEBUGOUT(" +------------------------------------------+\r\n");
  DEBUGOUT(" |      STARTING AUDIO CLASSIFIER           |\r\n");
  DEBUGOUT(" +------------------------------------------+\r\n");
  DEBUGOUT("\r\n");

  DEBUGOUT(" [ML] Loading model...\r\n");
  if (load_model() != SL_STATUS_OK) {
    DEBUGOUT(" [ML] ERROR: Model load failed!\r\n");
    return;
  }
  DEBUGOUT(" [ML] Model loaded OK\r\n");

  // Speaker init here — AFTER ML clocks are fully configured.
  // This is the only safe place. ULP UART needs full clock setup first.
  if (!speaker_initialized) {
    DEBUGOUT(" [BOOT] Init Speaker (DFMini)...\r\n");
    speaker_init();
    speaker_initialized = true;
    DEBUGOUT(" [BOOT] Speaker    : OK\r\n");
  }

  DEBUGOUT("\r\n");
  DEBUGOUT(" ==========================================\r\n");
  DEBUGOUT(" |   SYSTEM READY - MONITORING ACTIVE     |\r\n");
  DEBUGOUT(" |                                        |\r\n");
  DEBUGOUT(" |  >> Press button  to trigger alert     |\r\n");
  DEBUGOUT(" |  >> Speak keyword to trigger alert     |\r\n");
  DEBUGOUT(" |                                        |\r\n");
  DEBUGOUT(" |  Wake word: assist                     |\r\n");  // [WAKE WORD] updated
  DEBUGOUT(" |  Keywords : food, water, restroom      |\r\n");
  DEBUGOUT(" |             pain, emergency             |\r\n");  // [WAKE WORD] "assist" moved to wake
  DEBUGOUT(" ==========================================\r\n");
  DEBUGOUT("\r\n");

  while (1) {
    button_task();

    const uint32_t current_timestamp =
        sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

    int elapsed = current_timestamp - prev_loop_timestamp;
    if (elapsed < INFERENCE_INTERVAL_MS) {
      sl_sleeptimer_delay_millisecond(INFERENCE_INTERVAL_MS - elapsed);
    }

    command_recognizer->base_timestamp_ = current_timestamp;
    prev_loop_timestamp = current_timestamp;

    sl_ml_audio_feature_generation_update_features();

    const bool should_run_inference =
        (!SL_ML_FRONTEND_ACTIVITY_DETECTION_ENABLE ||
         (sl_ml_audio_feature_generation_activity_detected() == SL_STATUS_OK));

    if (should_run_inference) {
      if (run_inference() != SL_STATUS_OK) break;
    }

    process_output(should_run_inference);
  }
}

sl_status_t run_inference()
{
  sl_status_t status =
      sl_ml_audio_feature_generation_fill_tensor(
        keyword_spotting_model.input());
  if (status != SL_STATUS_OK) return SL_STATUS_FAIL;

  if (slx_ml_keyword_spotting_model_run() != SL_STATUS_OK)
    return SL_STATUS_FAIL;

  auto& output_tensor = *keyword_spotting_model.output(0);

  return SL_STATUS_OK;
}

sl_status_t process_output(const bool did_run_inference)
{
  uint8_t      result         = 0;
  uint8_t      score          = 0;
  bool         is_new_command = false;
  sl_status_t  status         = SL_STATUS_OK;
  TfLiteStatus process_status = kTfLiteOk;

  const uint32_t current_timestamp =
      sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());

  if (did_run_inference)
    process_status = command_recognizer->ProcessLatestResults(
        keyword_spotting_model.output(), current_timestamp,
        &result, &score, &is_new_command);

  if (process_status == kTfLiteOk)
    handle_results(current_timestamp, result, score, is_new_command);
  else
    status = SL_STATUS_FAIL;

  return status;
}

void handle_results(int32_t current_time, int result,
                    uint8_t score, bool is_new_command)
{
  const char *label = get_category_label(result);

  if (is_new_command) {
    if (SUPPRESION_TIME_MS <= 1) {
      sl_ml_audio_feature_generation_reset();
    }

    if (!button_active) {

      uint8_t track = get_track_for_label(label);
      if (track > 0) {
        DEBUGOUT("\r\n");
        DEBUGOUT(" +------------------------------------------+\r\n");
        DEBUGOUT(" |       PATIENT REQUEST  [ML VOICE]        |\r\n");
        DEBUGOUT(" +------------------------------------------+\r\n");
        DEBUGOUT(" |  Keyword : %-30s|\r\n", label);
        DEBUGOUT(" |  Score   : %-4d                          |\r\n", score);
        DEBUGOUT(" |  Track   : %d                             |\r\n", track);
        DEBUGOUT(" |  Time    : %-10ldms                  |\r\n", current_time);
        DEBUGOUT(" +------------------------------------------+\r\n");
        DEBUGOUT("\r\n");

        speaker_play(track);
      } else {
        DEBUGOUT(" [INFO] Unknown keyword: %s\r\n", label);
      }
    }

#ifdef SI917_DEVKIT
    led_set_color(DETECTION_LED, green_color);
    led_turn_on(DETECTION_LED);
#else
    led_turn_on(DETECTION_LED);
    led_turn_off(ACTIVITY_LED);
#endif

    activity_timestamp = 0;
    detected_timeout   = current_time + 1200;

  }
  else if (detected_timeout != 0 && current_time >= detected_timeout) {

    detected_timeout         = 0;
    previous_score           = score;
    previous_result          = result;
    previous_score_timestamp = current_time;

#ifdef SI917_DEVKIT
    led_set_color(ACTIVITY_LED, red_color);
#else
    led_turn_off(DETECTION_LED);
#endif
  }

  // 🔥 Removed wake word logic completely

  if (SL_ML_FRONTEND_ACTIVITY_DETECTION_ENABLE) {
    if (detected_timeout == 0 && score > 0 && previous_result != result) {
      activity_timestamp = current_time + 1000;
    } else if (current_time >= activity_timestamp) {
      activity_timestamp = 0;
#ifdef SI917_DEVKIT
      led_set_color(ACTIVITY_LED, red_color);
#else
      led_turn_off(ACTIVITY_LED);
#endif
    }

    if (activity_timestamp != 0) {
      if (current_time - activity_toggle_timestamp >= 100) {
        activity_toggle_timestamp = current_time;
#ifdef SI917_DEVKIT
        led_set_color(ACTIVITY_LED, red_color);
#endif
        led_toggle(ACTIVITY_LED);
      }
    }
    return;
  }

  if (detected_timeout == 0) {
    if (previous_score == 0) {
      previous_result          = result;
      previous_score           = score;
      previous_score_timestamp = current_time;
      return;
    }

    const int32_t time_delta  = current_time - previous_score_timestamp;
    const int8_t  score_delta = (int8_t)(score - previous_score);
    const float   diff        =
        (time_delta > 0) ? std::fabs(score_delta) / time_delta : 0.0f;

    previous_score           = score;
    previous_score_timestamp = current_time;

    if (diff >= SENSITIVITY ||
        (previous_result != result && score > 255 / 2)) {
      previous_result    = result;
      activity_timestamp = current_time + 500;
    } else if (current_time >= activity_timestamp) {
      activity_timestamp = 0;
#ifdef SI917_DEVKIT
      led_set_color(ACTIVITY_LED, red_color);
#else
      led_turn_off(ACTIVITY_LED);
#endif
    }

    if (activity_timestamp != 0) {
      if (current_time - activity_toggle_timestamp >= 100) {
        activity_toggle_timestamp = current_time;
#ifdef SI917_DEVKIT
        led_set_color(ACTIVITY_LED, red_color);
#endif
        led_toggle(ACTIVITY_LED);
      }
    }
  }
}
const char* get_category_label(int index)
{
  if ((index >= 0) && (index < category_count))
    return CATEGORY_LABELS[index];
  return "?";
}

sl_status_t load_model()
{
  auto accelerator = register_tflite_micro_accelerator();
  DEBUGOUT(" [ML] Accelerator : %s\r\n", accelerator->name());

  if (slx_ml_keyword_spotting_model_init() != SL_STATUS_OK) {
    DEBUGOUT(" [ML] ERROR: Model init failed\r\n");
    return SL_STATUS_FAIL;
  }

  if (!sl_ml_audio_feature_generation_load_model_settings(
         keyword_spotting_model_flatbuffer)) {
    DEBUGOUT(" [ML] ERROR: AFG settings failed\r\n");
    return SL_STATUS_FAIL;
  }

  if (!audio_classifier_config_load_model_settings(
         keyword_spotting_model_flatbuffer)) {
    DEBUGOUT(" [ML] ERROR: App config failed\r\n");
    return SL_STATUS_FAIL;
  }

  const TfLiteTensor *input  = keyword_spotting_model.input();
  const TfLiteTensor *output = keyword_spotting_model.output();

  if ((output->dims->size == 2) && (output->dims->data[0] == 1)) {
    category_count = output->dims->data[1];
  } else {
    DEBUGOUT(" [ML] ERROR: Invalid output tensor shape\r\n");
    return SL_STATUS_FAIL;
  }

  if (category_count != static_cast<int>(SL_TFLITE_MODEL_CLASSES.size())) {
    DEBUGOUT(" [ML] ERROR: Category mismatch %d != %ld\r\n",
             category_count, SL_TFLITE_MODEL_CLASSES.size());
    return SL_STATUS_FAIL;
  }

  if (!(input->type == kTfLiteInt8   ||
        input->type == kTfLiteUInt16 ||
        input->type == kTfLiteFloat32)) {
    DEBUGOUT(" [ML] ERROR: Invalid input tensor type\r\n");
    return SL_STATUS_FAIL;
  }

  if (!(output->type == kTfLiteInt8 ||
        output->type == kTfLiteFloat32)) {
    DEBUGOUT(" [ML] ERROR: Invalid output tensor type\r\n");
    return SL_STATUS_FAIL;
  }

  if (sl_ml_audio_feature_generation_init() != SL_STATUS_OK) {
    DEBUGOUT(" [ML] ERROR: AFG init failed\r\n");
    return SL_STATUS_FAIL;
  }

  command_recognizer = new (command_recognizer_instance_buffer)
      RecognizeCommands(SMOOTHING_WINDOW_DURATION_MS,
                        DETECTION_THRESHOLD,
                        SUPPRESION_TIME_MS,
                        MINIMUM_DETECTION_COUNT,
                        IGNORE_UNDERSCORE_LABELS);

  DEBUGOUT(" [ML] Categories  : %d\r\n", category_count);
  DEBUGOUT(" [ML] Model ready.\r\n");

  return SL_STATUS_OK;
}
