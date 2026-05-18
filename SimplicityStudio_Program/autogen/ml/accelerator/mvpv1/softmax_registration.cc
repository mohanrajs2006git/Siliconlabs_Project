#include "tflite_micro_model_config.h"


#include "ml/accelerator/mvpv1/mvpv1_kernel_registration_helper.hpp"

// NOTE: The MVPv1 does not support the Softmax kernel so we directly call the CMSIS-NN implementation

namespace tflite {

extern int          cmsis_nn_softmax_get_init_size();
extern TfLiteStatus cmsis_nn_softmax_prepare(TfLiteContext* context, TfLiteNode* node);
extern TfLiteStatus cmsis_nn_softmax_invoke(TfLiteContext* context, TfLiteNode* node);


namespace
{
using namespace npu_toolkit;

void* init(TfLiteContext* context, const char* buffer, size_t length)
{
    return registration_init(
        context,
        buffer,
        length,
        cmsis_nn_softmax_get_init_size()
    );
}


}  // namespace


TFLMRegistration Register_SOFTMAX() {
    return tflite::micro::RegisterOp(
        init,
        cmsis_nn_softmax_prepare,
        cmsis_nn_softmax_invoke
    );
}

}  // namespace tflite
