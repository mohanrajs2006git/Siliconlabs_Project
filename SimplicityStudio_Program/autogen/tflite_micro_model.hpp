#include "tflite_micro_model_config.h"
#pragma once
#include <cstdint>

#include "ml/utils/flags_helper.hpp"
#include "ml/third_party/tflm/micro_interpreter.h"
#include "ml/third_party/tflm/micro_op_resolver.h"

#include "ml/tflite_micro_model/tflite_micro_logger.hpp"
#include "ml/tflite_micro_model/tflite_model_parameters.hpp"
#include "ml/tflite_micro_model/tflite_micro_tensor.hpp"
#include "ml/tflite_micro_model/tflite_micro_accelerator.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_status.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_context.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_types.hpp"


#ifdef TFLITE_MICRO_PROFILER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_profiler.hpp"
#endif
#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "ml/tflite_micro_model/tflite_micro_recorder.hpp"
#endif // #ifdef TFLITE_MICRO_RECORDER_ENABLED

namespace npu_toolkit
{
class TfliteMicroPython;



class TfliteMicroModel
{
public:
  TfliteModelParameters parameters;

  /**
   * Default constructor
   */
  TfliteMicroModel();

  /**
   * Cleanup any allocated data
   */
  ~TfliteMicroModel();

  /**
   * @brief Load model flatbuffer
   *
   * Load a model flatbuffer (.tflite).
   *
   * @note The provided `flatbuffer` MUST persist for the life of this model object.
   *
   * @note If runtime_buffer is NULL then a buffer will be automatically allocated. In this case,
   * - If runtime_buffer_size <= 0, then automatically find the optimal run-time buffer size
   * - If runtime_buffer_size > 0, then allocate the specified size, if the size is too small then return the error
   *
   * @param flatbuffer Model flatbuffer (.tflite) binary data
   * @param op_resolver @ref tflite::MicroOpResolver with reigstered kernels
   * @param runtime_buffer Buffer to hold model working memory
   * @param runtime_buffer_size Size of the given runtime_buffer in bytes
   * @return true if model successfully loaded, false else
   */
  bool load(
    const void* flatbuffer,
    const tflite::MicroOpResolver& op_resolver,
    uint8_t *runtime_buffer = nullptr,
    int32_t runtime_buffer_size = 0
  );

  /**
   * @brief Load model with multiple runtime memory buffers
   *
   * Load a model flatbuffer (.tflite) with multiple runtime memory buffers.
   * The model must be pre-compiled to leverage the given memory buffers.
   *
   * @note The provided `flatbuffer` MUST persist for the life of this model object.
   *
   * The first entry in the buffer list is also used to store "persistent" memory
   * and any temporary allocations.
   *
   * @note If buffer[0] is NULL then a buffer will be automatically allocated. In this case,
   * - If buffer_sizes[0] <= 0, then automatically find the optimal run-time buffer size
   * - If buffer_sizes[0] > 0, then allocate the specified size, if the size is too small then return the error
   *
   * @param flatbuffer Model flatbuffer (.tflite) binary data
   * @param op_resolver @ref tflite::MicroOpResolver with reigstered kernels
   * @param buffers List of buffers to hold the model's runtime buffers
   * @param buffer_sizes Size of each buffer
   * @param n_buffers Number of buffer provided
   * @return true if model successfully loaded, false else
   */
  bool load(
    const void* flatbuffer,
    const tflite::MicroOpResolver* op_resolver,
    uint8_t* const buffers[],
    const int32_t buffer_sizes[],
    int32_t n_buffers
  );

  /**
   * @brief Unload model
   *
   * Unload model and clean up and allocated resources
   */
  void unload();

  /**
   * @brief Return if a model is loaded
   *
   * @return Return true if a model was successfully loaded, false otherwise
   */
  bool is_loaded() const;

  /**
   * Reset the state to be what you would expect when the interpreter is first
  * created. i.e. after Init and Prepare is called for the very first time.
  */
  bool reset();

  /**
   * @brief Invoke model inference
   *
   * Execute the loaded model
   *
   * @return true if model executed successfully, false else
   */
  bool invoke();


  /**
   * @brief Return number of input tensors
   *
   * @return The number of model input tensors
   */
  unsigned n_inputs() const;

  /**
   * @brief Get input tensor
   *
   * Populate the provided @ref TfliteTensorView with the
   * details of the input tensor at the given index.
   *
   * @param index Optional, index of input tensor
   * @return @ref TfliteTensorView to populate with input tensor at `index`
   */
  TfliteTensorView* input(unsigned index = 0) const;

  /**
   * @brief Return number of output tensors
   *
   * @return The number of model output tensors
   */
  unsigned n_outputs() const;

  /**
   * @brief Get output tensor
   *
   * Populate the provided @ref TfliteTensorView with the
   * details of the output tensor at the given index.
   *
   * @param index Optional, index of output tensor
   * @return @ref TfliteTensorView to populate with output tensor at `index`
   */
  TfliteTensorView* output(unsigned index = 0) const;

  /**
   * @brief Find metadata with tag in flatbuffer
   *
   * Find metadata with the given `tag` in the model's flatubffer
   *
   * @param tag Tag to search for in flatbuffer's metadata
   * @param length Optional, pointer to hold length of metadata's binary data
   * @return Pointer to found metadata's buffer in flatbuffer, null if not found
   */
  const void* find_metadata(const char* tag, uint32_t* length = nullptr) const;

  /**
   * Return a pointer to .tflite flatbuffer loaded by the model
   */
  const void* flatbuffer() const;

  /**
   *  Return a pointer to the TfliteMicroInterpreter
   * used by the model
   */
  tflite::MicroInterpreter* interpreter() const;
  /**
   * Return a pointer to the tflite::MicroOpResolver
   * used by the model
   */
  const tflite::MicroOpResolver* ops_resolver() const;

  /**
   * Return a pointer to the TfliteContext of the MicroInterpreter instance
   * used by the model
   */
  TfLiteContext* tflite_context() const;

  /**
   * Return a reference to the TfliteMicroModelStatus
   * used by the model
   */
  TfliteMicroModelStatus* status();

  /**
   * Return a pointer to the TfliteMicroAccelerator
   * used by the model
   */
  TfliteMicroAccelerator* accelerator(bool allow_null=false) const;

  /**
   * Return a pointer to the tflite::MicroAllocator
   * used by the model
   */
  tflite::MicroAllocator* allocator(bool allow_null=false) const;

  /**
   * Set a callback to be invoked before and after each layer of the model
   * This must be set BEFORE the model is loaded
  */
  bool set_layer_callback(
    TfliteMicroLayerCallback callback,
    void* arg = nullptr
  );

  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  /**
   * Enable/Disable profiling of the ML model
   *
   * @note This must be called BEFORE the model is loaded
   */
  bool set_profiler_enabled(bool enable);

  /**
   * Return if profiling is enabled
   *
   * @return true if profiler is enabled, false else
   */
  bool is_profiler_enabled() const;

  /**
   * Return the root model profiler
   *
   * @return Pointer to this model's profiler instance
   */
  profiling::Profiler* root_profiler() const;

  /**
   * Return a reference to the TfliteMicroProfiler
   * used by this model
   */
  TfliteMicroProfiler* profiler() const;


  static void profiling_metrics_callback(
    const profiling::Profiler* profiler,
    logging::Logger* logger,
    void *arg
  );
  #endif // TFLITE_MICRO_PROFILER_ENABLED

  /**
   * Return the GIT repository version
   * of Tensorflow-Lite Micro that this API uses.
   */
  static const char* tflite_micro_version();

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  using RecordingFlags = TfliteMicroRecorder::RecordingFlags;

  /**
   * Enable recording of model information.
   *
   * @note This must be called BEFORE the model is loaded
   *
   * @return true if recorder enabled, false else
   */
  bool set_recording_flags(RecordingFlags flags = RecordingFlags::Basic);

  /**
   * Return the flags used for recoding the model
   *
   * @return RecordingFlags
   */
  RecordingFlags get_recording_flags() const;

  /**
   * Return the recorded data from the previous inference
   * The returned data is msgpack formatted.
   */
  bool get_recorded_data(
    const uint8_t** buffer_ptr,
    uint32_t* length_ptr
  );

  TfliteMicroRecorder* recorder() const;
  #endif // TFLITE_MICRO_RECORDER_ENABLED


protected:
  uint8_t _interpreter_buffer[sizeof(tflite::MicroInterpreter)];
  TfliteMicroModelStatus _status;
  TfliteMicroAccelerator* _accelerator = nullptr;
  tflite::MicroAllocator* _allocator = nullptr;
  tflite::MicroInterpreter* _interpreter = nullptr;
  const tflite::MicroOpResolver* _ops_resolver = nullptr;
  const void* _flatbuffer = nullptr;
  uint8_t* _runtime_buffer = nullptr;
  int32_t* _buffer_sizes = nullptr;
  int8_t _n_buffers = 0;

  #ifdef TFLITE_MICRO_RECORDER_ENABLED
  TfliteMicroRecorder* _recorder = nullptr;
  #define _TFLITE_MICRO_RECORDER_CB 1
  #else
  #define _TFLITE_MICRO_RECORDER_CB 0
  #endif
  #ifdef TFLITE_MICRO_PROFILER_ENABLED
  TfliteMicroProfiler* _profiler = nullptr;
  #define _TFLITE_MICRO_PROFILER_CB 1
  #else
  #define _TFLITE_MICRO_PROFILER_CB 0
  #endif
  #ifdef NPU_TOOLKIT_WRAPPER_BUILD_ENABLED
  void* python_kernels = nullptr;
  #endif

  #ifdef TFLITE_MICRO_CAPTURE_MODEL_STATUS_MESSAGES
  #define _TFLITE_MICRO_STATUS_CB 1
  #else
  #define _TFLITE_MICRO_STATUS_CB 0
  #endif

  cpputils::ArrayHelper<
    TfliteMicroSectionCallbackContext,
   _TFLITE_MICRO_RECORDER_CB + _TFLITE_MICRO_PROFILER_CB
  > _section_callbacks;

  cpputils::ArrayHelper<
    TfliteMicroLayerCallbackContext,
    1 + _TFLITE_MICRO_RECORDER_CB + _TFLITE_MICRO_PROFILER_CB + _TFLITE_MICRO_STATUS_CB
  > _layer_callbacks;

  bool add_section_callback(
    TfliteMicroSectionCallback callback,
    void *arg = nullptr
  );
  bool remove_section_callback(
    TfliteMicroSectionCallback callback,
    void *arg = nullptr
  );
  bool add_layer_callback(
    TfliteMicroLayerCallback callback,
    void *arg = nullptr
  );
  bool remove_layer_callback(
    TfliteMicroLayerCallback callback,
    void *arg = nullptr
  );
  TfLiteStatus process_section_callbacks(
    TfliteMicroSection section,
    bool is_start
  );

  TfLiteStatus process_layer_callbacks(
    TfliteMicroSection section,
    bool is_start,
    int index,
    const tflite::NodeAndRegistration& node_and_registration,
    TfLiteStatus execution_status = kTfLiteOk
  );

  bool load_interpreter(
    const void* flatbuffer,
    const tflite::MicroOpResolver* op_resolver,
    uint8_t* buffers[],
    const int32_t buffer_sizes[],
    int32_t n_buffers
  );
  bool create_allocator(
    const void* flatbuffer,
    uint8_t* const buffers[],
    const int32_t buffer_sizes[],
    int32_t n_buffers
  );

  #ifdef TFLITE_MICRO_DYNAMIC_TENSOR_BUFFER_ALLOCATION_ENABLED
  bool find_optimal_buffer_size(
    const void* flatbuffer,
    const tflite::MicroOpResolver* op_resolver,
    uint8_t* buffers[],
    int32_t buffer_sizes[],
    int32_t n_buffers,
    int32_t &optimal_buffer_size
  );
  #endif

  /// @cond NO_DOXYGEN
  friend TfliteMicroProfiler;
  friend TfliteMicroRecorder;
  friend TfliteMicroPython;
  friend TfliteMicroModelStatus;
  friend tflite::MicroInterpreter;
  friend tflite::MicroInterpreterGraph;
  /// @endcond
};

} // namespace npu_toolkit
