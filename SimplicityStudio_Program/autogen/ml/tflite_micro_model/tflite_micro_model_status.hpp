#include "tflite_micro_model_config.h"
#pragma once

#include "ml/third_party/tflm/micro_allocator.h"
#include "ml/third_party/tflm/tls-schema_generated.h"

#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_types.hpp"



#if (defined(TFLITE_MICRO_LOGGER_ENABLED) && TFLITE_MICRO_LOG_LEVEL <= 3) || defined(TFLITE_MICRO_RECORDER_ENABLED)
#define TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
#endif



namespace npu_toolkit
{

class TfliteMicroRecorder;
class TfliteMicroModel;


class TfliteMicroModelStatus
{
public:
  TfliteMicroModelStatus() = default;

  static TfliteMicroModelStatus& instance();

  void init(TfliteMicroModel *model);
  void set_op_error_reporter(struct TfLiteContext* context);

  void reset();
  void issue(bool is_fatal, const char* fmt, ...);
  void vissue(bool is_fatal, const char* fmt, va_list args);
  void flush();


  bool is_fatal_error_detected() const
  {
    return _fatal_error_detected;
  }
  bool is_warning_detected() const
  {
    return _warning_detected;
  }

  static void TfliteReportOpError(
    struct TfLiteContext* context,
    const char* format,
    ...
  );

private:
  TfliteMicroModel* _model = nullptr;
  bool _fatal_error_detected = false;
  bool _warning_detected = false;

  static TfLiteStatus _layer_callback(
      TfliteMicroSection section,
    bool is_start,
    int index,
    const tflite::NodeAndRegistration& node_and_registration,
    TfLiteStatus status,
    void* arg
  );

  #ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
  #ifdef __arm__
  static constexpr const int MSG_BUFFER_SIZE = 128;
  #else
  static constexpr const int MSG_BUFFER_SIZE = 4096;
  #endif
  char _msg_buffer[MSG_BUFFER_SIZE] = { 0 };
  char *_msg_ptr = nullptr;

  int remaining_message_buffer_length() const;
  int used_message_buffer_length() const;
  void append_msg(const char* s);

  #endif // TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES

};

extern "C" void npu_toolkit_issue_model_warning(const char* msg, ...);


} // namespace npu_toolkit




#ifndef NPU_TOOLKIT_ISSUE_MODEL_WARNING
  #define NPU_TOOLKIT_ISSUE_MODEL_WARNING_(status, fmt, ...) status.issue(false, fmt, ## __VA_ARGS__)
  #define NPU_TOOLKIT_ISSUE_MODEL_WARNING(fmt, ...) \
    NPU_TOOLKIT_ISSUE_MODEL_WARNING_(::npu_toolkit::TfliteMicroModelStatus::instance(), fmt, ## __VA_ARGS__)
#endif

#ifndef NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR
  #define NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR_(status, fmt, ...) status.issue(true, fmt, ## __VA_ARGS__)
  #define NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR(fmt, ...) \
    NPU_TOOLKIT_ISSUE_MODEL_FATAL_ERROR_(::npu_toolkit::TfliteMicroModelStatus::instance(), fmt, ## __VA_ARGS__)
#endif


#ifndef NPU_TOOLKIT_RESET_MODEL_ERRORS
#define NPU_TOOLKIT_RESET_MODEL_ERRORS() \
  ::npu_toolkit::TfliteMicroModelStatus::instance().reset()
#endif
