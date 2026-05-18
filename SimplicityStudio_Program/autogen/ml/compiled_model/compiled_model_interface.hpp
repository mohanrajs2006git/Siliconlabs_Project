#include "tflite_micro_model_config.h"
#pragma once


#include <cstdint>
#include "ml/utils/memory_util.hpp"


namespace npu_toolkit
{


constexpr const char COMPILED_MODEL_DATA_TFLITE_TAG[] = "sl_model_data_v1";

/**
 * The context mode used by the embedded device
  processing the compiled data
*/
enum class CompiledContextMode : int8_t
{
    Direct = 0, //!< Directly read the compiled context from the .tflite model's metadata
    Buffered = 1, //!< Copy the entire compiled context from the .tflite model's metadata to a RAM buffer
    Cached = 2, //!< Page the compiled context metadata from the .tflite to a RAM cache buffer as needed
    Invalid = -1
};

/**
 * An id used to indicate which accelerator
 * is used by the CompiledLayerConfig
 *
 * Refer to the corresponding Python:
 * npu_toolkit.compiler.common.embedded_inferface.CompiledAcceleratorId
*/
enum class CompiledAcceleratorId : int8_t
{
  NoAccelerator = 0, //!< No accelerator used by layer
  Mvp = 1, //!< MVPv1 or 2
  MvpTse = 2, //!< MVPv2 + TSE
  Gce = 3, //!< GCE
};


/**
 * A generic wrapper for compiled data structures
 *
 * The first HEADER_LENGTH bytes contains the length in bytes of the compiled
 * item (including the bytes containing the length).
 *
 * The rest of the data is dependent on the data struct
 * that was compiled.
 *
 * Refer to the corresponding Python:
 * npu_toolkit.compiler.common.embedded_inferface_common.CompiledItem
 */
template<unsigned HEADER_LENGTH>
struct CompiledItem
{
  /**
   * The length in bytes of the length header
   */
  static constexpr const int HEADER_LENGTH_BYTES = HEADER_LENGTH;


  /**
   * Return a pointer to the beginning of thge underlying data.
   * If offset=0 (default) then this points to the HEADER_LENGTH bytes.
   */
  const uint8_t* base(int offset = 0) const
  {
    return reinterpret_cast<const uint8_t*>(this) + offset;
  }

  /**
   * The length in bytes of this compiled item.
   * The length includes the HEADER_LENGTH-byte length, e.g.:
   * length_bytes = HEADER_LENGTH + len(data struct)
   */
  uint32_t length_bytes() const
  {

    if constexpr(HEADER_LENGTH == 2)
    {
      // The first 2 bytes is a 16-bit value containing the number of 32-bit words
      return bytes_to_uint16(base(0)) * sizeof(uint32_t);
    }
    else if constexpr(HEADER_LENGTH == 4)
    {
      // The first 4 bytes is a 32-bit value containing the number of 32-bit words
      return bytes_to_uint32(base(0)) * sizeof(uint32_t);
    }
    else
    {
      static_assert(HEADER_LENGTH ==2 || HEADER_LENGTH == 4);
    }

  }

  /**
   * Pointer to the beginning of the underlying data.
   */
  const uint8_t* data(int offset = 0) const
  {
    return base(HEADER_LENGTH + offset);
  }

  /**
   * Pointer to the end of the data.
   * data_end = data() + length_bytes()
   */
  const uint8_t* data_end() const
  {
    return base(length_bytes());
  }

};

/**
 * Container to holder 1 or more CompiledItems
 *
 * NOTE: This container data struct plus the CompiledItems
 * data struct allows for caching data from external NVM
 * to an internal SRAM buffer.
 *
 * Refer to the corresponding Python:
 * npu_toolkit.compiler.common.embedded_inferface_common.CompiledContainer
*/
template<typename T>
struct CompiledContainer
{
  static constexpr const int HEADER_LENGTH_BYTES = 4;

  static const CompiledContainer<T>* create(const void* p)
  {
    return reinterpret_cast<const CompiledContainer<T>*>(p);
  }

  // The number of CompiledContainer following the current
  uint8_t n_subsequent_containers() const
  {
    return *data(0);
  }

  /**
   * The header of a container is its **unaligned length** in bytes as a 24-bit int.
   * NOTE: While this length is NOT 32-bit aligned, the CompiledContainer packed data is
   * (i.e the data packed into the CompiledContainer is padded with 0s to be 32-bit aligned)
  */
  uint32_t length_bytes() const
  {
    return bytes_to_uint24(data(1));
  }

  /**
   * The 32-bit aligned length of this container
   */
  uint32_t aligned_length_bytes() const
  {
    return align_up_uint32(length_bytes());
  }

  /**
   * Return a pointer to this container's data
   */
  const uint8_t* data(int offset = 0) const
  {
    return reinterpret_cast<const uint8_t*>(this) + offset;
  }

  /**
   * Return a pointer to the end of this container's data
   */
  const uint8_t* data_end() const
  {
    return data(length_bytes());
  }

  /**
   * Return a pointer to the first CompiledItem in this container
   */
  const T* first_item() const
  {
    return reinterpret_cast<const T*>(data(sizeof(uint32_t)));
  }

  /**
   * Create a container with the given buffer
   */
  static CompiledContainer<T>* create(
    uint8_t* buffer,
    uint8_t n_subsequent_containers,
    uint32_t length
  )
  {
    buffer[0] = n_subsequent_containers;
    buffer[1] = (length >> 0) & 0xff;
    buffer[2] = (length >> 8) & 0xff;
    buffer[3] = (length >> 16) & 0xff;
    return reinterpret_cast<CompiledContainer<T>*>(buffer);
  }
};


/**
 * Contains the compressed program info for a single program:
 *
 * This has the format:
 * <# register groups><register offsets><padding><register values>
 *
 * Where <register offsets> has the format:
 * <# registers group0><group0 offsets><# registers group1><group1 offsets>...
 *
 * <padding> has format:
 * Pad with 0 to ensure <register values> is 32-bit algined
 *
 * <register values>  has format:
 * <group0 32-bit register values><group1 32-bit register values>...
 *
 * Refer to the corresponding Python:
 * npu_toolkit.compiler.register_compressor.embedded_inferface.CompressedProgramConfig
*/
struct CompressedProgramConfig : CompiledItem<sizeof(uint16_t)>
{
  /**
   * The number of register groups
   */
  uint8_t n_register_groups() const
  {
    return *data(0);
  }

  /**
   * Pointer to the register offsets
   */
  const uint8_t* register_offsets() const
  {
    return data(1);
  }

  /**
   * Pointer to register values
   */
  const uint32_t* register_values() const
  {
    const uint8_t n_register_groups = this->n_register_groups();
    const uint8_t* offsets = register_offsets();
    // Iterate to the end of the register offsets
    for(int i = 0; i < n_register_groups; ++i)
    {
      const uint8_t n_registers = *offsets++;
      offsets += n_registers;
    }

    // Get the 32-bit aligned address which points to the register values
    const uintptr_t unaligned_addr = (uintptr_t)offsets;
    const uintptr_t aligned_addr = align_up_uint32(unaligned_addr);
    return (const uint32_t*)aligned_addr;
  }

  /**
   * Return a pointer to the next CompressedProgramConfig in the list
   */
  const CompressedProgramConfig* next() const
  {
    return reinterpret_cast<const CompressedProgramConfig*>(data_end());
  }
};


/**
 * Contains all the compressed program info for a single model layer
 *
 * This has the following format:
 * <# programs><packed CacheLayerConfig or 0><padding><list of packed CompressedProgramConfig>
 *
 * Refer to the corresponding Python:
 * npu_toolkit.compiler.common.embedded_inferface.CompiledLayerConfig
*/
struct CompiledLayerConfig : CompiledItem<sizeof(uint16_t)>
{
  /**
   * The number of accelerator programs used by this layer
  */
  uint16_t n_programs() const
  {
    return bytes_to_uint16(data(0));
  }

  /**
   * Return a pointer to the CacheLayerConfig or nullptr if not available
   */
  const uint8_t* cache_layer_config() const
  {
    const uint32_t cache_config_len = bytes_to_uint16(data(2));
    return (cache_config_len != 0) ? data(2) : nullptr;
  }

  /**
   * Return a pointer to the first compressed program
   */
  const CompressedProgramConfig* first_program() const
  {
    const uint32_t cache_config_len = bytes_to_uint16(data(2));
    const uint8_t* unaligned_addr = (cache_config_len == 0) ? data(2 + 2) : data(2 + cache_config_len);
    const uintptr_t aligned_addr = align_up_uint32((uintptr_t)unaligned_addr);
    return (const CompressedProgramConfig*)aligned_addr;
  }

    /**
   * Return a pointer to the next CompiledLayerConfig in the list
   */
  const CompiledLayerConfig* next() const
  {
    return reinterpret_cast<const CompiledLayerConfig*>(data_end());
  }
};


/**
 * Contains all compilation data for a model
 *
 * This has the format:
 * <# layers><layer accelerator ids><context mode><padding><layer configs><cache config><cache xfrs>
 *
 * Refer to the corresponding Python:
 * npu_toolkit.compiler.common.embedded_inferface.CompiledModelData
*/
struct CompiledModelData : CompiledItem<sizeof(uint32_t)>
{
  /**
   * Wrap the given bytes with this data struct
   */
  static const CompiledModelData* create(const void* p)
  {
    return reinterpret_cast<const CompiledModelData*>(p);
  }

  /*
   * The number of layers in the model
  */
  uint16_t n_layers() const
  {
    return bytes_to_uint16(data(0));
  }

  /**
   * Return a list of CompiledAcceleratorId,
   * which indicates the accelerator used by each layer of the model.
  */
  const CompiledAcceleratorId* layer_accelerators() const
  {
    return reinterpret_cast<const CompiledAcceleratorId*>(data(2));
  }

  /**
   * Return the mode in which the model context should be retrieved
   */
  CompiledContextMode context_mode() const
  {
    return (CompiledContextMode)(*data(2 + n_layers()));
  }

  /**
   * Return a pointer to the CompiledContainer<CompiledLayerConfig> data
  */
  const uint8_t* layer_config_ptr() const
  {
    const uint32_t unaligned_offset = 2 + n_layers() + 1;
    const uint32_t aligned_offset = align_up_uint32(unaligned_offset);
    return data(aligned_offset);
  }


  /**
   * Return a pointer to the CacheConfig data
   * if available, nullptr otherwise
  */
  const uint8_t* cache_config_ptr() const
  {
    auto ptr = this->layer_config_ptr();
    for(int i = 0; i <= 256; ++i)
    {
      assert(i < 256);
      const auto container = CompiledContainer<uint8_t>::create(ptr);
      ptr += container->aligned_length_bytes();
      if(container->n_subsequent_containers() == 0)
      {
        break;
      }
    }

    const uint32_t cache_config_len = bytes_to_uint16(ptr)*sizeof(uint32_t);
    return (cache_config_len > 0) ? ptr : nullptr;
  }

  /**
   * Return a pointer to the CompiledContainer<CacheTransferConfig> data
   * if available, nullptr otherwise
  */
  const uint8_t* cache_transfer_list_ptr() const
  {
    const uint8_t* cache_config_ptr = this->cache_config_ptr();
    if(cache_config_ptr == nullptr)
    {
      return nullptr;
    }
    // The first two bytes of the CacheConfig struct
    // is its uint16 length in 32-bit words
    const uint32_t cache_config_len = bytes_to_uint16(cache_config_ptr)*sizeof(uint32_t);

    // The transfer list comes immediately after the CacheConfig
    return cache_config_ptr + cache_config_len;
  }

};



} // namespace npu_toolkit