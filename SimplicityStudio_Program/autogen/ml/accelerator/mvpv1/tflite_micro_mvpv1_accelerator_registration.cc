#include "tflite_micro_model_config.h"
#include "ml/accelerator/mvpv1/tflite_micro_mvpv1_accelerator.hpp"




namespace npu_toolkit
{


static TfliteMicroMvpv1Accelerator accelerator;

extern TfliteMicroAccelerator* get_tflite_micro_accelerator()
{
  return &accelerator;
}

extern TfliteMicroAccelerator* register_tflite_micro_accelerator()
{
    return register_tflite_micro_accelerator(get_tflite_micro_accelerator());
}



} // namespace npu_toolkit