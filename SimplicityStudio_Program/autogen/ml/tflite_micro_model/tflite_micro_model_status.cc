#include "tflite_micro_model_config.h"
#include <cstdio>

#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "tflite_micro_model.hpp"
#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
#include "stacktrace.hpp"
#endif
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recorder.hpp"
#endif

namespace npu_toolkit
{

TfliteMicroModelStatus& TfliteMicroModelStatus::instance()
{
  auto model = TfliteMicroModelHelper::tflite_micro_model();
  return *model->status();
}

void TfliteMicroModelStatus::init(TfliteMicroModel *model)
{
  _model = model;
  #ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
  model->add_layer_callback(TfliteMicroModelStatus::_layer_callback, this);
  #endif
  reset();
}


void TfliteMicroModelStatus::reset()
{
  _fatal_error_detected = false;
  _warning_detected = false;
  #ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
  _msg_ptr = _msg_buffer;
  #endif
}

void TfliteMicroModelStatus::issue(bool is_fatal, const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  TfliteMicroModelStatus::vissue(is_fatal, fmt, args);
  va_end(args);
}

void TfliteMicroModelStatus::vissue(bool is_fatal, const char* fmt, va_list args)
{
  _warning_detected = true;
  if(!_fatal_error_detected)
  {
    _fatal_error_detected = is_fatal;
  }

  #ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
  char msg_buffer[MSG_BUFFER_SIZE];
  vsnprintf(msg_buffer, sizeof(msg_buffer), fmt, args);
  if(used_message_buffer_length())
  {
    append_msg(". ");
  }
  append_msg(msg_buffer);
  #endif
}


TfLiteStatus TfliteMicroModelStatus::_layer_callback(
  TfliteMicroSection section,
  bool is_start,
  int index,
  const tflite::NodeAndRegistration& node_and_registration,
  TfLiteStatus status,
  void* arg
)
{
  auto& self = *reinterpret_cast<TfliteMicroModelStatus*>(arg);
  if(is_start)
  {
    return status;
  }

  self.flush();

  return status;
}

void TfliteMicroModelStatus::flush()
{
  #ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
  if(used_message_buffer_length() == 0)
  {
    return;
  }

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  auto recorder = _model->recorder();
  if(recorder != nullptr)
  {
    auto msgpack = recorder->_msgpack;
    msgpack_write_dict_dict(msgpack, "error_msg", 2);
    msgpack_write_dict_bool(msgpack, "is_fatal", _fatal_error_detected);
    msgpack_write_dict_str(msgpack, "msg", _msg_buffer);
  }
  #endif // TFLITE_MICRO_RECORDER_ENABLED

  #ifdef TFLITE_MICRO_LOGGER_ENABLED
  bool printed_msg = false;
  auto context = TfliteMicroModelHelper::get_active_context(true);

  if(context != nullptr)
  {
    const auto model_context = TfliteMicroModelHelper::tflite_micro_model_context(context);

    if(model_context != nullptr && model_context->current_layer_index != -1)
    {
      const auto layer_name = TfliteMicroModelHelper::current_layer_name(context);
      char log_buffer[MSG_BUFFER_SIZE + 128];
      snprintf(log_buffer, sizeof(log_buffer), "%s: %s", layer_name, _msg_buffer);
      get_logger().write_buffer(_fatal_error_detected ? logging::Error : logging::Warn, log_buffer);
      printed_msg = true;
    }
  }

  if(!printed_msg)
  {
    get_logger().write_buffer(_fatal_error_detected ? logging::Error : logging::Warn, _msg_buffer);
  }
  #endif // TFLITE_MICRO_LOGGER_ENABLED

  _msg_ptr = _msg_buffer;

  #endif // TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
}


#ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES

int TfliteMicroModelStatus::remaining_message_buffer_length() const
{
  if(_msg_ptr == nullptr)
  {
    return 0;
  }
  return (int)(&_msg_buffer[MSG_BUFFER_SIZE-1] - _msg_ptr);
}

int TfliteMicroModelStatus::used_message_buffer_length() const
{
  if(_msg_ptr == nullptr)
  {
    return 0;
  }
  return (int)(_msg_ptr - _msg_buffer);
}

void TfliteMicroModelStatus::append_msg(const char* s)
{
  if(_msg_ptr == nullptr)
  {
    return;
  }

  int remaining = remaining_message_buffer_length();
  while(*s != 0 && remaining > 0)
  {
    *_msg_ptr++ = *s++;
    remaining--;
  }
  *_msg_ptr = 0;
}

#endif // TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES

void TfliteMicroModelStatus::set_op_error_reporter(struct TfLiteContext* context)
{
  context->ReportError = TfliteMicroModelStatus::TfliteReportOpError;
}
void TfliteMicroModelStatus::TfliteReportOpError(
  struct TfLiteContext* context,
  const char* format,
  ...
)
{
  va_list args;
  va_start(args, format);
  TfliteMicroModelStatus::instance().vissue(true, format, args);
  va_end(args);

  #if defined(TFLITE_MICRO_SIMULATOR_ENABLED) && !defined(NDEBUG)
  stacktrace_print_stack();
  #endif

}

} // namespace npu_toolkit

extern "C" void npu_toolkit_issue_model_warning(const char* msg, ...)
{
  va_list args;
  va_start(args, msg);
  ::npu_toolkit::TfliteMicroModelStatus::instance().issue(false, msg, args);
  va_end(args);
}