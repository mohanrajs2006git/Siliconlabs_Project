#include "tflite_micro_model_config.h"
#pragma once

#include <cstring>
#include <stdint.h>
#include <stdarg.h>
#include <functional>

#include "ml/utils/typed_list.hpp"
#include "ml/utils/flags_helper.hpp"



namespace logging
{

class Logger;

constexpr const unsigned MAX_TAG_LENGTH = 15;
constexpr const unsigned MAX_FMT_LENGTH = 256;

enum class Level : uint8_t
{
    Debug,
    Info,
    Warn,
    Error,
    Disabled,
    Count
};
constexpr Level Debug = Level::Debug;
constexpr Level Info = Level::Info;
constexpr Level Warn = Level::Warn;
constexpr Level Error = Level::Error;
constexpr Level Disabled = Level::Disabled;


enum class Flag : uint8_t
{
    None        = 0,
    PrintTag   = (1 << 0),
    PrintLevel = (1 << 1),
    Newline     = (1 << 2)
};
DEFINE_ENUM_CLASS_BITMASK_OPERATORS(Flag, uint8_t)

constexpr Flag None = Flag::None;
constexpr Flag PrintTag = Flag::PrintTag;
constexpr Flag PrintLevel = Flag::PrintLevel;
constexpr Flag Newline = Flag::Newline;



typedef cpputils::FlagsHelper<Flag> Flags;

using Writer = std::function<void(const char*, int, void*)>;

class Logger
{
public:
    Logger(
        const char* tag = "",
        Level level = Level::Info,
        Flags flags = Flag::Newline
    );

    bool level(Level level);
    bool level(const char* level);
    bool level_is_enabled(Level level) const;
    const char* level_str(void) const;
    Flags flags(const Flags& flags);
    void writer(const Writer& writer, void *arg = nullptr);
    const Writer& writer() const;

    void debug(const char *fmt, ...) const;
    void info(const char *fmt, ...) const;
    void warn(const char *fmt, ...) const;
    void error(const char *fmt, ...) const;
    void dump_hex(const void * address, unsigned length, const char *desc=nullptr, ...);
    void vdump_hex(const void * address, unsigned length, const char *desc, va_list args);

    void write(Level level, const char *fmt, ...) const;
    void vwrite(Level level, const char *fmt, va_list args) const;
    void write_buffer(Level level, const char *buffer, int length = -1) const;

    Flags& flags(void)
    {
        return _flags;
    }

    const Flags& flags(void) const
    {
        return _flags;
    }

    constexpr Level level(void) const
    {
        return _level;
    }

    constexpr const char* tag() const
    {
        return _tag;
    }

protected:
    Level _level = Level::Info;
    Flags _flags;
    Writer _writer = nullptr;
    void* _writer_arg = nullptr;
    char _tag[MAX_TAG_LENGTH + 1] = "";

    int write_prefix(Level level, char *buffer, unsigned buffer_length) const;
};


} // namespace logging

