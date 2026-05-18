#include "tflite_micro_model_config.h"
#pragma once


namespace cpputils
{

/**
 * @brief Flags (i.e. bitfield) helper
 */
template<typename FlagsType>
struct FlagsHelper
{
    FlagsType value;

    FlagsHelper(FlagsType value = (FlagsType)0) : value(value)
    {
    }

    void write(FlagsType v)
    {
        value = v;
    }

    constexpr FlagsType read() const
    {
        return value;
    }

    constexpr FlagsType read(FlagsType mask) const
    {
        return value & mask;
    }

    void set(FlagsType flags)
    {
        value |= flags;
    }

    void set(FlagsType flags, FlagsType mask)
    {
        value = (value & ~mask) | flags;
    }

    void set_value(FlagsType flags, bool value)
    {
        if(value)
        {
            set(flags);
        }
        else
        {
            clear(flags);
        }
    }

    void clear(FlagsType flags)
    {
        value &= ~flags;
    }

    constexpr bool some_set() const
    {
        return value != (FlagsType)0;
    }

    constexpr bool some_set(FlagsType flags) const
    {
        return (value & flags) != (FlagsType)0;
    }

    constexpr bool is_set(FlagsType flags) const
    {
        return (value & flags) != (FlagsType)0;
    }

    constexpr bool all_set(FlagsType flags) const
    {
        return (value & flags) == flags;
    }

    FlagsHelper(const FlagsHelper &other) : value(other.value)
    {
    }

    FlagsHelper& operator= (const FlagsHelper &other)
    {
        value = other.value;
        return *this;
    }

    FlagsHelper& operator= (const FlagsType type)
    {
        value = type;
        return *this;
    }

    bool operator == (const FlagsType other) const
    {
        return value == other;
    }

    bool operator == (const FlagsHelper &other) const
    {
        return value == other.value;
    }

    bool operator != (const FlagsType other) const
    {
        return value != other;
    }

    bool operator != (const FlagsHelper &other) const
    {
        return value != other.value;
    }

    FlagsHelper& operator |= (const FlagsHelper &other)
    {
        value |= other.value;
        return *this;
    }

    FlagsHelper& operator |= (const FlagsType type)
    {
        value |= type;
        return *this;
    }

    FlagsHelper& operator &= (const FlagsHelper &other)
    {
        value &= other.value;
        return *this;
    }

    FlagsHelper& operator &= (const FlagsType type)
    {
        value &= type;
        return *this;
    }

    FlagsHelper& operator ^= (const FlagsHelper &other)
    {
        value ^= other.value;
        return *this;
    }

    FlagsHelper& operator ^= (const FlagsType type)
    {
        value ^= type;
        return *this;
    }

    FlagsHelper operator| (const FlagsHelper &other) const
    {
        return value | other.value;
    }

    FlagsHelper operator| (const FlagsType type) const
    {
        return value | type;
    }

    FlagsHelper operator& (const FlagsHelper &other) const
    {
        return value & other.value;
    }

    FlagsHelper operator& (const FlagsType type) const
    {
        return value & type;
    }

    FlagsHelper operator^ (const FlagsHelper &other) const
    {
        return value ^ other.value;
    }

    FlagsHelper operator^ (const FlagsType type) const
    {
        return value ^ type;
    }

    FlagsHelper operator~ () const
    {
        return ~value;
    }

    FlagsHelper operator! () const
    {
        return !value;
    }

};



} // namespace cpputils
