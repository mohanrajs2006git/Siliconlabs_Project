#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>

#include "ml/utils/helpers.hpp"
#include "ml/compiled_model/compiled_model_interface.hpp"


namespace npu_toolkit
{

/**
 * The maximum number of ping-pong buffer groups
*/
constexpr const int CACHE_MAX_BUFFER_GROUP_COUNT  = 8;
/**
 * The total number of buffers
*/
constexpr const int CACHE_MAX_BUFFER_COUNT  = CACHE_MAX_BUFFER_GROUP_COUNT*2;



/**
 * This is an arbitrary offset added to the MVP->ARRAY.ADDRCFG register.
 * This ensures the relative offsets are not negative.
 * The model compiler adds this value to the calculated MVP->ARRAY.ADDRCFG values
 * and the firmware subtracts this value from the MVP->ARRAY.BASE_OFFSET register in
 * the @ref CompiledModelCache::begin_layer() API.
*/
constexpr const uint32_t CACHE_BUFFER_ADDR_OFFSET  = 0x1000000UL;

/**
 * Cache Buffer Type
  i.e. What a particular cache buffer group is used for
*/
enum class CacheBufferType : uint8_t
{
  ModelReadyOnly = 0,     ///< Buffer is used for caching data from the model's weights and biases
  NonPersistent = 1,      ///< Buffer is used for caching non-persistent data (e.g. activations)
  PlannedPersistent = 2,  ///< Buffer is used for caching planned-persistent data
  LayerConfig = 3,        ///< Buffer is used for caching the CacheLayerConfig list chunks
  TransferList = 4        ///< Buffer is used for caching the CacheTransferConfig list chunks
};

/**
 * The buffer data direction
*/
enum class CacheBufferDirection : uint8_t
{
  Output = 0,   ///< e.g. Cache buffer -> External memory (e.g. PSRAM)
  Input = 1     ///< e.g. External memory -> cache buffer
};

/**
 * The lifetime of a buffer
 * i.e. How long the data within the buffer is valid
*/
enum class CacheBufferLifetime : uint8_t
{
  Unused = 0,               ///< The buffer is not used
  ReleaseAfterUse = 1,      ///< The buffer is released immediately after it is used
  ReleaseAtEndOfLayer = 2,  ///< The buffer is released at the end of the currently executing model layer
  ManualRelease = 3         ///< The buffer is manually released as configured in the CacheLayerConfig
};



/**
 * Cache Buffer Group configuration
 * This is used during CompiledModelCache::init()
 * to initialize a cache buffer "group"
 *
 * A buffer "group" is two ping-pong cache buffers.
*/
#pragma pack(push, 1)
struct CacheBufferGroupConfig
{
  uint8_t memory_region : 4;  ///< Memory region id
                              ///< i.e. Memory region where data will be copied from or to, e.g.
                              ///< Input: External memory region -> cache buffer
                              ///< Output: Cache buffer -> PSRAM memory region
  CacheBufferType type : 3;   ///< The type of data this buffer stores
  uint8_t auto_wrap : 1;      ///< Flag to enable auto-wrapping for this buffer
};
#pragma pack(pop)


/**
 * Defines the layout for a specific cache buffer
 * This includes the memory region of the cache buffer,
 * and the offset from the beginning of the memory region
 * for the cache buffer's base address.
*/
#pragma pack(push, 1)
struct CacheBufferOffset
{
  uint32_t offset  : 28;      ///< Offset, in bytes, from the beginning of the memory region
  uint32_t memory_region : 4; ///< The memory region id where the buffer resides
};
#pragma pack(pop)


/**
 * DMA transfer configuration
*/
struct CacheTransferConfig
{
  static constexpr const int LENGTH_BYTES = 8;

  /**
   * A valid layer id is between 0-254
   * We use 255 indicate that the layer id is not set
   */
  static constexpr const uint8_t NULL_LAYER_ID = 255;

  const uint8_t* data(int offset = 0) const
  {
    return reinterpret_cast<const uint8_t*>(this) + offset;
  }

  /**
   * The size of this data struct in bytes
  */
  constexpr uint32_t length_bytes() const
  {
    return CacheTransferConfig::LENGTH_BYTES;
  }

  const CacheTransferConfig* next() const
  {
    return reinterpret_cast<const CacheTransferConfig*>(data(CacheTransferConfig::LENGTH_BYTES));
  }

  /**
   * 24-bit byte offset from the beginning of the source or destination memory region
   *
  */
  uint32_t offset() const
  {
    return bytes_to_uint24(data(0));
  }

  /**
   * The number (16-bits) of 32-bit words to transfer
  */
  uint32_t count() const
  {
    return bytes_to_uint16(data(3));
  }

  /**
   * Id of cache buffer to copy from/to
  */
  uint8_t buffer_id() const
  {
    return *data(5) & 0x0F;
  }

  /**
   * The direction of this transfer
   * Input: External memory region -> cache buffer
   * Output: Cache buffer -> external memory region
  */
  CacheBufferDirection direction() const
  {
    const uint8_t v = *data(5) & 0x80;
    return (v != 0) ? CacheBufferDirection::Input : CacheBufferDirection::Output;
  }

  /**
   * Optional layer id of this transfer.
   * If this is NOT 255, then the transfer will not execute
   * until the corresponding layer is active
  */
  uint8_t layer_id() const
  {
    return *data(6);
  }

  static CacheTransferConfig* create(
    uint8_t* buffer,
    uint32_t offset,
    uint16_t count,
    uint8_t buffer_id,
    CacheBufferDirection direction,
    uint8_t layer_id
  )
  {
    buffer[0] = (offset >> 0) & 0xff;
    buffer[1] = (offset >> 8) & 0xff;
    buffer[2] = (offset >> 16) & 0xff;
    buffer[3] = (count >> 0) & 0xff;
    buffer[4] = (count >> 8) & 0xff;
    buffer[5] = buffer_id | ((int)direction << 7);
    buffer[6] = layer_id;
    buffer[7] = 0;

    return reinterpret_cast<CacheTransferConfig*>(buffer);
  }

};


/**
 * Cache Configuration
 *
 * This is used during CompiledModelCache::init()
 * to initialize the cache.
*/
struct CacheConfig
{
  static const CacheConfig* create(const void* p)
  {
    return reinterpret_cast<const CacheConfig*>(p);
  }

  const uint8_t* data(int offset = 0) const
  {
    return reinterpret_cast<const uint8_t*>(this) + offset;
  }

  /**
   * The size in bytes of this data struct
  */
  uint32_t length_bytes() const
  {
    return bytes_to_uint16(data(0))*sizeof(uint32_t);
  }

  /**
   * The number of buffer groups
   * A buffer "group" is two ping-pong buffers
  */
  uint8_t n_buffer_groups() const
  {
    return *data(2);
  }

  /**
   * Return the CacheBufferGroupConfig for a specific buffer group
  */
  const CacheBufferGroupConfig buffer_group_config(int group_id) const
  {
    return *reinterpret_cast<const CacheBufferGroupConfig*>(data(3 + group_id));
  }

  /**
   * Return the CacheBufferOffset for a specific buffer
  */
  const CacheBufferOffset buffer_offset(int buffer_id) const
  {
    const uint8_t offset = \
      3 +
      n_buffer_groups()*sizeof(CacheBufferGroupConfig) +
      buffer_id*sizeof(CacheBufferOffset);
    return *reinterpret_cast<const CacheBufferOffset*>(data(offset));
  }
};


/***
 * Optional flags for each CacheLayerConfig
 */
enum class CacheLayerConfigFlag : uint8_t
{
  None = 0,
  DelayedStart = (1 << 0) ///< Caching for this layer is delayed until a delimiter is encounter in the compressed program register offsets
};
DEFINE_ENUM_CLASS_OPERATORS(CacheLayerConfigFlag, uint8_t);

/**
 * Cache configuration for an individual model layer
*/
struct CacheLayerConfig
{
  static const CacheLayerConfig* create(const void* p)
  {
    return reinterpret_cast<const CacheLayerConfig*>(p);
  }

  const uint8_t* data(int offset = 0) const
  {
    return reinterpret_cast<const uint8_t*>(this) + offset;
  }

  /**
   * The lifetime of each cache buffer used by this layer
  */
  CacheBufferLifetime buffer_lifetime(int group_id) const
  {
    if(group_id < 4)
    {
      const uint8_t group_config = *data(0);
      return (CacheBufferLifetime)((group_config >> (group_id*2)) & 0x03);
    }
    else
    {
      const uint8_t group_config = *data(1);
      group_id -= 4;
      return (CacheBufferLifetime)((group_config >> (group_id*2)) & 0x03);
    }
  }

  /**
   * Bitmask for each buffers' direction
   * 0 -> output
   * 1 -> input
  */
  uint16_t buffer_direction_mask() const
  {
    return bytes_to_uint16(data(2));
  }

  /**
   * Bitmask indicating which buffers
   * this layer must wait to be released before executing.
   *
   * 1 -> wait for buffer id at corresponding bit index
   * 0 -> don't wait for buffer
  */
  uint16_t initial_wait_mask() const
  {
    return bytes_to_uint16(data(4));
  }

  /**
   * Bitmask indicating which buffers
   * this layer must wait to be populated before executing.
   *
   * 1 -> wait for buffer id at corresponding bit index
   * 0 -> don't wait for buffer
  */
  uint16_t final_wait_mask() const
  {
    return bytes_to_uint16(data(6));
  }

  /**
   * Optional flags for this layer
   */
  CacheLayerConfigFlag flags() const
  {
    return (CacheLayerConfigFlag)(*data(8));
  }

  /**
   * The number of MVP arrays this layer uses for caching
  */
  uint8_t n_mvp_array_ids() const
  {
    return *data(9);
  }

  /**
   * Get the MVP array id at the given index
  */
  uint8_t mvp_array_id(int index) const
  {
    const auto array_info = *data(10 + index);
    return array_info & 0x0F;
  }

  /**
   * Return the memory region used by the given MVP array id
  */
  uint8_t mvp_array_memory_region(int index) const
  {
    const auto array_info = *data(10 + index);
    return array_info >> 4;
  }

  /**
   * List of manual release flags used by this layer
  */
  const uint16_t* manual_release_flags() const
  {
    const uint8_t aligned_offset = align_up_uint16(10 + n_mvp_array_ids());
    return reinterpret_cast<const uint16_t*>(data(aligned_offset));
  }

  /**
   * Helper function to access an individual manual_wait_mask flag
  */
  static uint16_t manual_wait_mask(const uint16_t* flags)
  {
    return (flags == nullptr) ? 0 : *flags;

  }

  /**
   * Helper function to access an individual manual_release_mask flag
  */
  static uint16_t manual_release_mask(const uint16_t* flags)
  {
    if(flags == nullptr)
    {
      return 0;
    }
    const uint16_t current_wait_mask = flags[0];
    const uint16_t next_wait_mask = flags[1];
    return (current_wait_mask ^ next_wait_mask) & current_wait_mask;
  }
};


} // namespace npu_toolkit