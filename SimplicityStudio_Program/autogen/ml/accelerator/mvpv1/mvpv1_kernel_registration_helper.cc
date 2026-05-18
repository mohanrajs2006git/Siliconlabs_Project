#include "tflite_micro_model_config.h"
#include <algorithm>
#include "ml/accelerator/mvpv1/mvpv1_kernel_registration_helper.hpp"

namespace npu_toolkit
{


/**
 * This is helper to update the user_data of TfLiteNode
 * when entering and exiting a fallback implementation.
 *
 * The opdata struct contains a KernelType as its first member.
 * The accelerator registration uses this to determine if the
 * layer is supported by the accelerator.
 *
 * If the layer is not supported, we revert to the fallback
 * implementation which uses its own data struct that does not include
 * a KernelType.
 *
 * So, this helper simple updates te user_data pointer
 * to not include the KernelType before entering the fallback implementation.
 */
struct FallbackNode
{
  TfLiteNode* node;
  FallbackNode(TfLiteNode* node) : node(node)
  {
    node->user_data = (uint8_t*)node->user_data + sizeof(KernelType);
  }

  ~FallbackNode()
  {
    node->user_data = (uint8_t*)node->user_data - sizeof(KernelType);
  }

};


void* registration_init(
  TfLiteContext* context,
  const char* buffer,
  size_t length,
  int init_size,
  int fallback_init_size
)
{
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  //return

  int alloc_size = 0;
  if(fallback_init_size != 0)
  {
    fallback_init_size += sizeof(KernelType);
  }

  // If the current layer is disabled then use the fallback implementation
  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  if(TfliteMicroAccelerator::get_current_layer_disabled(context))
  {
    // Then use the fallback's init size
    alloc_size = fallback_init_size;
  }
  #endif

  #ifdef MVPV1_COMPILED_KERNELS_ENABLED
  // If the model is compiled
  if(npu_toolkit::mvpv1_is_model_compiled(context))
  {
    // Is the current layer compiled (i.e. is it  supported by the accelerator)?
    if(npu_toolkit::mvpv1_is_current_layer_compiled(context))
    {
      // It is, so we use that acc impl init size
      alloc_size = init_size;
    }
    else
    {
      // Otherwise, the current layer is not supported,
      // So, use the fallback implementation's init size
      alloc_size = fallback_init_size;
    }
  }
  #endif // MVPV1_COMPILED_KERNELS_ENABLED

  if(alloc_size == 0)
  {
    // Otherwise, the model has not been compiled or compiled kernels aren't enabled
    // In this case, we must use the maximum between the accelerator's init size
    // and the fallback's init size
    alloc_size = std::max(init_size, fallback_init_size);
  }

  // Allocate a buffer for the given size and return a pointer to it
  return context->AllocatePersistentBuffer(context, alloc_size);
}


TfLiteStatus registration_prepare(
  TfLiteContext* context,
  TfLiteNode* node,
  KernelPrepare prepare,
  KernelPrepare fallback_prepare
)
{
  // If the current layer is disabled then use the fallback implementation
  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  if(TfliteMicroAccelerator::get_current_layer_disabled(context))
  {
    return (fallback_prepare != nullptr) ? fallback_prepare(context, node) : kTfLiteError;
  }
  #endif

  #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
  // If the model is compiled
  if(npu_toolkit::mvpv1_is_model_compiled(context))
  {
    // Is the current layer is compiled (i.e. is it supported by the accelerator)?
    if(npu_toolkit::mvpv1_is_current_layer_compiled(context))
    {
        // It is, so just use the mvp implementation
        return prepare(context, node);
    }
    // Otherwise use the fallback implementation if it's available
    else if(fallback_prepare != nullptr)
    {
      FallbackNode fallback_node(node);
      return fallback_prepare(context, node);
    }
    else
    {
      // Otherwise, return an error
      return kTfLiteError;
    }
  }
  #endif // MVPV1_COMPILED_KERNELS_ENABLED


  // Otherwise, the model has not been compiled or compiled kernels aren't enabled
  #ifndef MVPV1_COMPILED_KERNELS_ONLY
  // Call the accelerator's prepare implementation
  auto status = prepare(context, node);

  // If the accelerator's prepare method returns success,
  // then this layer is supported by the accelerator.
  if(status == kTfLiteOk)
  {
    return kTfLiteOk;
  }
  // If the accelerator's prepare method returns a delegate error,
  // then that means this layer is not supported by the accelerator.
  // In this case, try to use the fallback implementation
  else if(status == kTfLiteDelegateError)
  {
    if(fallback_prepare != nullptr)
    {
      FallbackNode fallback_node(node);
      return fallback_prepare(context, node);
    }
    else
    {
      // Otherwise, return an error
      return kTfLiteError;
    }
  }
  #endif // MVPV1_COMPILED_KERNELS_ONLY

  // In all other cases, return an error
  return kTfLiteError;
}

TfLiteStatus registration_invoke(
  TfLiteContext* context,
  TfLiteNode* node,
  KernelInvoke invoke,
  KernelInvoke fallback_invoke
)
{
  // If the current layer is disabled then use the fallback implementation
  #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
  if(TfliteMicroAccelerator::get_current_layer_disabled(context))
  {
    return (fallback_invoke != nullptr) ? fallback_invoke(context, node) : kTfLiteError;
  }
  #endif

  // If the current layer is compiled
  #if defined(MVPV1_COMPILED_KERNELS_ENABLED)
  // If the model is compiled
  if(npu_toolkit::mvpv1_is_model_compiled(context))
  {
    // Is the current layer is compiled (i.e. is it supported by the accelerator)?
    if(npu_toolkit::mvpv1_is_current_layer_compiled(context))
    {
        // It is, so just use the mvp implementation
        return invoke(context, node);
    }
    // Otherwise use the fallback implementation if it's available
    else if(fallback_invoke != nullptr)
    {
      FallbackNode fallback_node(node);
      return fallback_invoke(context, node);
    }
    else
    {
      // Otherwise, return an error
      return kTfLiteError;
    }
  }
  #endif // MVPV1_COMPILED_KERNELS_ENABLED

  // Otherwise, the current layer has not been compiled or compiled kernels aren't enabled
  #ifndef MVPV1_COMPILED_KERNELS_ONLY

  // Invoke the accelerator's implementation
  auto status = invoke(context, node);

  // If the accelerator implementation returns a delegate error,
  // then that means this layer is not supported by the accelerator.
  // In this case, use the fallback implementation
  if(status == kTfLiteDelegateError)
  {
    // If we have a fallback implementation then use that
    if(fallback_invoke != nullptr)
    {
      FallbackNode fallback_node(node);
      return fallback_invoke(context, node);
    }
    else
    {
      // Otherwise, return an error
      return kTfLiteError;
    }
  }
  return status;

  #else // MVPV1_COMPILED_KERNELS_ONLY
  return kTfLiteError;
  #endif // MVPV1_COMPILED_KERNELS_ONLY
}



} //namespace npu_toolkit