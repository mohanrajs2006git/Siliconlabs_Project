#pragma once
/* This was generated from:
Path: C:/Users/Mohan Raj S/SimplicityStudio/v5_workspace/EIC2025/config/tflite/keyword_spotting.tflite
+-------+-------------------+-----------------+----------------+-------------------------------------------------------------------+
| Index | OpCode            | Input(s)        | Output(s)      | Config                                                            |
+-------+-------------------+-----------------+----------------+-------------------------------------------------------------------+
| 0     | conv_2d           | 98x1x104 (int8) | 98x1x24 (int8) | Padding:Same stride:1x1 activation:None                           |
|       |                   | 3x1x104 (int8)  |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 1     | conv_2d           | 98x1x24 (int8)  | 98x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 2     | depthwise_conv_2d | 98x1x72 (int8)  | 49x1x72 (int8) | Multiplier:1 padding:Same stride:2x2 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 3     | conv_2d           | 49x1x72 (int8)  | 49x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 4     | conv_2d           | 98x1x24 (int8)  | 49x1x24 (int8) | Padding:Same stride:2x2 activation:Relu                           |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 5     | add               | 49x1x24 (int8)  | 49x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 49x1x24 (int8)  |                |                                                                   |
| 6     | conv_2d           | 49x1x24 (int8)  | 49x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 7     | depthwise_conv_2d | 49x1x72 (int8)  | 49x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 8     | conv_2d           | 49x1x72 (int8)  | 49x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 9     | add               | 49x1x24 (int8)  | 49x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 49x1x24 (int8)  |                |                                                                   |
| 10    | conv_2d           | 49x1x24 (int8)  | 49x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 11    | depthwise_conv_2d | 49x1x72 (int8)  | 49x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 12    | conv_2d           | 49x1x72 (int8)  | 49x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 13    | add               | 49x1x24 (int8)  | 49x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 49x1x24 (int8)  |                |                                                                   |
| 14    | conv_2d           | 49x1x24 (int8)  | 49x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 15    | depthwise_conv_2d | 49x1x72 (int8)  | 49x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 16    | conv_2d           | 49x1x72 (int8)  | 49x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 17    | add               | 49x1x24 (int8)  | 49x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 49x1x24 (int8)  |                |                                                                   |
| 18    | conv_2d           | 49x1x24 (int8)  | 49x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 19    | depthwise_conv_2d | 49x1x72 (int8)  | 25x1x72 (int8) | Multiplier:1 padding:Same stride:2x2 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 20    | conv_2d           | 25x1x72 (int8)  | 25x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 21    | conv_2d           | 49x1x24 (int8)  | 25x1x24 (int8) | Padding:Same stride:2x2 activation:Relu                           |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 22    | add               | 25x1x24 (int8)  | 25x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 25x1x24 (int8)  |                |                                                                   |
| 23    | conv_2d           | 25x1x24 (int8)  | 25x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 24    | depthwise_conv_2d | 25x1x72 (int8)  | 25x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 25    | conv_2d           | 25x1x72 (int8)  | 25x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 26    | add               | 25x1x24 (int8)  | 25x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 25x1x24 (int8)  |                |                                                                   |
| 27    | conv_2d           | 25x1x24 (int8)  | 25x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 28    | depthwise_conv_2d | 25x1x72 (int8)  | 25x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 29    | conv_2d           | 25x1x72 (int8)  | 25x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 30    | add               | 25x1x24 (int8)  | 25x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 25x1x24 (int8)  |                |                                                                   |
| 31    | conv_2d           | 25x1x24 (int8)  | 25x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 32    | depthwise_conv_2d | 25x1x72 (int8)  | 25x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 33    | conv_2d           | 25x1x72 (int8)  | 25x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 34    | add               | 25x1x24 (int8)  | 25x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 25x1x24 (int8)  |                |                                                                   |
| 35    | conv_2d           | 25x1x24 (int8)  | 25x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 36    | depthwise_conv_2d | 25x1x72 (int8)  | 13x1x72 (int8) | Multiplier:1 padding:Same stride:2x2 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 37    | conv_2d           | 13x1x72 (int8)  | 13x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 38    | conv_2d           | 25x1x24 (int8)  | 13x1x24 (int8) | Padding:Same stride:2x2 activation:Relu                           |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 39    | add               | 13x1x24 (int8)  | 13x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 13x1x24 (int8)  |                |                                                                   |
| 40    | conv_2d           | 13x1x24 (int8)  | 13x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 41    | depthwise_conv_2d | 13x1x72 (int8)  | 13x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 42    | conv_2d           | 13x1x72 (int8)  | 13x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 43    | add               | 13x1x24 (int8)  | 13x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 13x1x24 (int8)  |                |                                                                   |
| 44    | conv_2d           | 13x1x24 (int8)  | 13x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 45    | depthwise_conv_2d | 13x1x72 (int8)  | 13x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 46    | conv_2d           | 13x1x72 (int8)  | 13x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 47    | add               | 13x1x24 (int8)  | 13x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 13x1x24 (int8)  |                |                                                                   |
| 48    | conv_2d           | 13x1x24 (int8)  | 13x1x72 (int8) | Padding:Valid stride:1x1 activation:Relu                          |
|       |                   | 1x1x24 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 49    | depthwise_conv_2d | 13x1x72 (int8)  | 13x1x72 (int8) | Multiplier:1 padding:Same stride:1x1 dilation:1x1 activation:Relu |
|       |                   | 9x1x72 (int8)   |                |                                                                   |
|       |                   | 72 (float16)    |                |                                                                   |
| 50    | conv_2d           | 13x1x72 (int8)  | 13x1x24 (int8) | Padding:Valid stride:1x1 activation:None                          |
|       |                   | 1x1x72 (int8)   |                |                                                                   |
|       |                   | 24 (float16)    |                |                                                                   |
| 51    | add               | 13x1x24 (int8)  | 13x1x24 (int8) | Activation:Relu                                                   |
|       |                   | 13x1x24 (int8)  |                |                                                                   |
| 52    | average_pool_2d   | 13x1x24 (int8)  | 1x1x24 (int8)  | Padding:Valid stride:1x13 filter:1x13 activation:None             |
| 53    | reshape           | 1x1x24 (int8)   | 24 (int8)      | Type=reshapeoptions                                               |
|       |                   | 2 (int32)       |                |                                                                   |
| 54    | fully_connected   | 24 (int8)       | 9 (int8)       | Activation:None                                                   |
|       |                   | 24 (int8)       |                |                                                                   |
|       |                   | 9 (float16)     |                |                                                                   |
| 55    | softmax           | 9 (int8)        | 9 (int8)       | Beta:1.0                                                          |
+-------+-------------------+-----------------+----------------+-------------------------------------------------------------------+
*/

#define SL_TFLITE_MODEL_AVERAGE_WINDOW_DURATION_MS 300
#define SL_TFLITE_MODEL_CLASSES { "food", "water", "restroom", "emergency", "pain", "assist", "noise", "help", "_unknown_" }
#define SL_TFLITE_MODEL_DATE "2026-01-30T12:42:06.445Z"
#define SL_TFLITE_MODEL_DETECTION_THRESHOLD_LIST { 216, 216, 216, 234, 234, 234, 216, 216, 255 }
#define SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ALPHA_A 0.5f
#define SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ALPHA_B 0.800000011920929f
#define SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ARM_THRESHOLD 0.75f
#define SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_ENABLE false
#define SL_TFLITE_MODEL_FE_ACTIVITY_DETECTION_TRIP_THRESHOLD 0.800000011920929f
#define SL_TFLITE_MODEL_FE_DC_NOTCH_FILTER_COEFFICIENT 0.949999988079071f
#define SL_TFLITE_MODEL_FE_DC_NOTCH_FILTER_ENABLE true
#define SL_TFLITE_MODEL_FE_FFT_LENGTH 512
#define SL_TFLITE_MODEL_FE_FILTERBANK_LOWER_BAND_LIMIT 125.0f
#define SL_TFLITE_MODEL_FE_FILTERBANK_N_CHANNELS 104
#define SL_TFLITE_MODEL_FE_FILTERBANK_UPPER_BAND_LIMIT 7500.0f
#define SL_TFLITE_MODEL_FE_LOG_SCALE_ENABLE true
#define SL_TFLITE_MODEL_FE_LOG_SCALE_SHIFT 6
#define SL_TFLITE_MODEL_FE_NOISE_REDUCTION_ENABLE true
#define SL_TFLITE_MODEL_FE_NOISE_REDUCTION_EVEN_SMOOTHING 0.02500000037252903f
#define SL_TFLITE_MODEL_FE_NOISE_REDUCTION_MIN_SIGNAL_REMAINING 0.4000000059604645f
#define SL_TFLITE_MODEL_FE_NOISE_REDUCTION_ODD_SMOOTHING 0.05999999865889549f
#define SL_TFLITE_MODEL_FE_NOISE_REDUCTION_SMOOTHING_BITS 10
#define SL_TFLITE_MODEL_FE_PCAN_ENABLE false
#define SL_TFLITE_MODEL_FE_PCAN_GAIN_BITS 21
#define SL_TFLITE_MODEL_FE_PCAN_OFFSET 80.0f
#define SL_TFLITE_MODEL_FE_PCAN_STRENGTH 0.949999988079071f
#define SL_TFLITE_MODEL_FE_QUANTIZE_DYNAMIC_SCALE_ENABLE true
#define SL_TFLITE_MODEL_FE_QUANTIZE_DYNAMIC_SCALE_RANGE_DB 40.0f
#define SL_TFLITE_MODEL_FE_SAMPLE_LENGTH_MS 1000
#define SL_TFLITE_MODEL_FE_SAMPLE_RATE_HZ 16000
#define SL_TFLITE_MODEL_FE_WINDOW_SIZE_MS 30
#define SL_TFLITE_MODEL_FE_WINDOW_STEP_MS 10
#define SL_TFLITE_MODEL_HASH "bf931d01555c61bc3a74e7b59c9cea0e"
#define SL_TFLITE_MODEL_LATENCY_MS 10
#define SL_TFLITE_MODEL_MINIMUM_COUNT 2
#define SL_TFLITE_MODEL_NAME "sil_train"
#define SL_TFLITE_MODEL_RUNTIME_MEMORY_SIZE 40648
#define SL_TFLITE_MODEL_SUPPRESSION_MS 700
#define SL_TFLITE_MODEL_VERBOSE_MODEL_OUTPUT_LOGS false
#define SL_TFLITE_MODEL_VERSION 1
#define SL_TFLITE_MODEL_VOLUME_GAIN 0.0f
