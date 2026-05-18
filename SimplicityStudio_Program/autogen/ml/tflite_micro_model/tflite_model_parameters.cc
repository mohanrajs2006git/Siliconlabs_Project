#include "tflite_micro_model_config.h"
#include "ml/tflite_micro_model/tflite_model_parameters.hpp"
#include "ml/tflite_micro_model/tflite_micro_model_helper.hpp"

namespace npu_toolkit
{


using StringList = TfliteModelParameters::StringList;
using Int32List = TfliteModelParameters::Int32List;
using FloatList = TfliteModelParameters::FloatList;


const char TfliteModelParameters::METADATA_TAG[] = "SL_PARAMSv1";


bool TfliteModelParameters::load_from_tflite_flatbuffer(
    const void* flatbuffer,
    TfliteModelParameters& parameters,
    const char* tag
)
{
    tag = (tag != nullptr) ? tag : TfliteModelParameters::METADATA_TAG;
    const void* metadata = TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(
        flatbuffer,
        tag
    );

    if(metadata == nullptr)
    {
        return false;
    }

    if(!parameters.load(metadata))
    {
        return false;
    }

    return true;
}

bool TfliteModelParameters::load_from_tflite_flatbuffer(
    const void *flatbuffer,
    const char* tag
)
{
    tag = (tag != nullptr) ? tag : TfliteModelParameters::METADATA_TAG;
    const void* metadata = TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(
        flatbuffer,
        tag
    );

    if(metadata == nullptr)
    {
        return false;
    }

    if(!load(metadata))
    {
        return false;
    }

    return true;
}

bool TfliteModelParameters::load(const schema::Dictionary *fb_dictionary)
{
    if (fb_dictionary == nullptr || fb_dictionary->entries() == nullptr)
    {
        return false;
    }
    else
    {
        _fb_dictionary = fb_dictionary;

        return true;
    }
}

bool TfliteModelParameters::load(const void *flatbuffer)
{
    const auto fb_dictionary = schema::GetDictionary(flatbuffer);
    return load(fb_dictionary);
}


void TfliteModelParameters::unload()
{
    _fb_dictionary = nullptr;
}

TfliteModelParameters::TfliteModelParameters(const TfliteModelParameters& other)
{
    load(other._fb_dictionary);
}

TfliteModelParameters& TfliteModelParameters::operator=(const TfliteModelParameters& other)
{
    load(other._fb_dictionary);
    return *this;
}

const TfliteModelParameters::Value *TfliteModelParameters::get(const char *key) const
{
    const TfliteModelParameters::Value *retval = nullptr;

    if (_fb_dictionary != nullptr)
    {
        for (auto entry : *this)
        {
            if (strcmp(entry->key(), key) == 0)
            {
                retval = entry;
                break;
            }
        }
    }

    return retval;
}

bool TfliteModelParameters::contains(const char *key) const
{
    bool retval = false;

    if (_fb_dictionary != nullptr)
    {
        for (auto entry : *this)
        {
            if (strcmp(entry->key(), key) == 0)
            {
                retval = true;
                break;
            }
        }
    }

    return retval;
}

const TfliteModelParameters::Value *TfliteModelParameters::operator[](const char *key) const
{
    return get(key);
}

TfliteModelParameters::Iterator TfliteModelParameters::begin(void) const
{
    return TfliteModelParameters::Iterator(_fb_dictionary->entries()->begin());
}

TfliteModelParameters::Iterator TfliteModelParameters::end() const
{
    return TfliteModelParameters::Iterator(_fb_dictionary->entries()->end());
}

bool TfliteModelParameters::get(const char *key, const char *&value) const
{
    auto entry = get(key);
    if (entry != nullptr && entry->type() == schema::Value::str)
    {
        value = entry->str();
        return true;
    }
    else
    {
        return false;
    }
}

bool TfliteModelParameters::get(const char *key, const uint8_t *&value) const
{
    auto entry = get(key);
    if (entry != nullptr && entry->type() == schema::Value::bin)
    {
        value = entry->bin();
        return true;
    }
    else
    {
        return false;
    }
}

bool TfliteModelParameters::get(const char *key, StringList &value) const
{
    auto entry = get(key);
    if (entry != nullptr && entry->type() == schema::Value::str_list)
    {
        value = entry->str_list();
        return true;
    }
    else
    {
        return false;
    }
}

bool TfliteModelParameters::get(const char *key, Int32List &value) const
{
    auto entry = get(key);
    if (entry != nullptr && entry->type() == schema::Value::int32_list)
    {
        value = entry->int32_list();
        return true;
    }
    else
    {
        return false;
    }
}

bool TfliteModelParameters::get(const char *key, FloatList &value) const
{
    auto entry = get(key);
    if (entry != nullptr && entry->type() == schema::Value::float_list)
    {
        value = entry->float_list();
        return true;
    }
    else
    {
        return false;
    }
}

bool TfliteModelParameters::get(const char* key, const uint8_t* &data, uint32_t &length) const
{
    auto entry = get(key);

    if (entry == nullptr || entry->type() != schema::Value::bin)
    {
        return false;
    }
    else
    {
        length = entry->bin_length();
        data = entry->bin();
        return true;
    }
}

#ifdef TFLITE_MICRO_LOGGER_ENABLED
void TfliteModelParameters::print_summary(logging::Logger* logger) const
{
    auto& l = (logger != nullptr) ? *logger : get_logger(LoggerId::NpuToolkit);

    auto print_list = [](
        logging::Logger& l,
        const Value& val,
        void (*printer)(logging::Logger& l, const Value& val)
    )
    {
        const auto saved_flags = l.flags();
        l.flags(saved_flags & ~logging::Newline);
        l.info("%s=[", val.key());
        l.flags(logging::None);

        printer(l, val);

        if((saved_flags & logging::Newline) == logging::Newline)
        {
            l.flags(logging::Newline);
        }
        l.info("]");
        l.flags(saved_flags);
    };


    for(const auto& val : *this)
    {
        const auto dtype = val->type();
        const auto key = val->key();

        if(dtype == schema::Value::boolean)
        {
            l.info("%s=%s", key, val->boolean() ? "true" : "false");
        }
        else if(dtype == schema::Value::i8)
        {
            const auto v = val->i8();
            l.info("%s=%d", key, v);
        }
        else if(dtype == schema::Value::i16)
        {
            const auto v = val->i16();
            l.info("%s=%d", key, v);
        }
        else if(dtype == schema::Value::i32)
        {
            const auto v = val->i32();
            l.info("%s=%d", key, v);
        }
        else if(dtype == schema::Value::i64)
        {
            const auto v = val->i64();
            l.info("%s=%ld", key, v);
        }
        else if(dtype == schema::Value::u8)
        {
            const auto v = val->u8();
            l.info("%s=%u", key, v);
        }
        else if(dtype == schema::Value::u16)
        {
            const auto v = val->u16();
            l.info("%s=%u", key, v);
        }
        else if(dtype == schema::Value::u32)
        {
            const auto v = val->u32();
            l.info("%s=%u", key, v);
        }
        else if(dtype == schema::Value::u64)
        {
            const auto v = val->u64();
            l.info("%s=%lu", key, v);
        }
        else if(dtype == schema::Value::f32)
        {
            const auto v = val->f32();
            l.info("%s=%f", key, v);
        }
        else if(dtype == schema::Value::f64)
        {
            const auto v = val->f64();
            l.info("%s=%lf", key, v);
        }
        else if(dtype == schema::Value::str)
        {
            const auto v = val->str();
            l.info("%s=%s", key, v);
        }
        else if(dtype == schema::Value::str_list)
        {
            print_list(l, *val, [](logging::Logger& l, const Value& val)
            {
                const auto& ls = val.str_list();
                const int sz = ls.size();
                for(int i = 0; i < sz; ++i)
                {
                    l.info("%s", ls[i]);
                    if(i < sz-1)
                    {
                        l.info(", ");
                    }
                }
            });
        }
        else if(dtype == schema::Value::int32_list)
        {
            print_list(l, *val, [](logging::Logger& l, const Value& val)
            {
                const auto& ls = val.int32_list();
                const int sz = ls.size();
                for(int i = 0; i < sz; ++i)
                {
                    l.info("%d", ls[i]);
                    if(i < sz-1)
                    {
                        l.info(", ");
                    }
                }
            });
        }
        else if(dtype == schema::Value::float_list)
        {
            print_list(l, *val, [](logging::Logger& l, const Value& val)
            {
                const auto& ls = val.float_list();
                const int sz = ls.size();
                for(int i = 0; i < sz; ++i)
                {
                    l.info("%f", ls[i]);
                    if(i < sz-1)
                    {
                        l.info(", ");
                    }
                }
            });
        }
        else if(dtype == schema::Value::bin)
        {
            l.info("%s=<%d bytes>", key, val->bin_length());
        }
    }
}
#endif // #ifdef TFLITE_MICRO_LOGGER_ENABLED



TfliteModelParameters::Iterator::Iterator(EntryVectorIterator it) : it(it)
{
}

bool TfliteModelParameters::Iterator::operator!=(Iterator rhs)
{
    return it != rhs.it;
}

const TfliteModelParameters::Value *TfliteModelParameters::Iterator::operator*()
{
    auto entry = *it;
    return reinterpret_cast<const Value *>(entry);
}

void TfliteModelParameters::Iterator::operator++()
{
    ++it;
}

schema::Value TfliteModelParameters::Value::type(void) const
{
    return value_type();
}

const char *TfliteModelParameters::Value::key(void) const
{
    return schema::Entry::key()->c_str();
}

bool TfliteModelParameters::Value::boolean(void) const
{
    return value_as_boolean()->value();
}

int8_t TfliteModelParameters::Value::i8(void) const
{
    return value_as_i8()->value();
}

uint8_t TfliteModelParameters::Value::u8(void) const
{
    return value_as_u8()->value();
}

int16_t TfliteModelParameters::Value::i16(void) const
{
    return value_as_i16()->value();
}

uint16_t TfliteModelParameters::Value::u16(void) const
{
    return value_as_u16()->value();
}

int32_t TfliteModelParameters::Value::i32(void) const
{
    return value_as_i32()->value();
}

uint32_t TfliteModelParameters::Value::u32(void) const
{
    return value_as_u32()->value();
}

int64_t TfliteModelParameters::Value::i64(void) const
{
    return value_as_i64()->value();
}

uint64_t TfliteModelParameters::Value::u64(void) const
{
    return value_as_u64()->value();
}

float TfliteModelParameters::Value::f32(void) const
{
    return value_as_f32()->value();
}

double TfliteModelParameters::Value::f64(void) const
{
    return value_as_f64()->value();
}

const char *TfliteModelParameters::Value::str(void) const
{
    return value_as_str()->data()->c_str();
}

TfliteModelParameters::StringList TfliteModelParameters::Value::str_list(void) const
{
    return StringList(value_as_str_list()->data());
}

TfliteModelParameters::Int32List TfliteModelParameters::Value::int32_list(void) const
{
    return Int32List(value_as_int32_list()->data());
}

TfliteModelParameters::FloatList TfliteModelParameters::Value::float_list(void) const
{
    return FloatList(value_as_float_list()->data());
}

const uint8_t *TfliteModelParameters::Value::bin(void) const
{
    return value_as_bin()->data()->Data();
}

uint32_t TfliteModelParameters::Value::bin_length(void) const
{
    return value_as_bin()->data()->size();
}

StringList::StringList(const StringVector *vector) : vector(vector)
{
}

StringList::StringList(const StringList& other)
{
    this->vector = other.vector;
}

StringList& StringList::operator=(const StringList& other)
{
    this->vector = other.vector;
    return *this;
}

uint32_t StringList::size(void) const
{
    return (vector != nullptr) ? vector->size() : 0;
}

const char *StringList::operator[](unsigned index) const
{
    return (vector != nullptr) ? vector->GetAsString(index)->c_str() : nullptr;
}

StringList::Iterator StringList::begin(void) const
{
    return (vector != nullptr) ? StringList::Iterator(vector->begin()) : StringList::Iterator();
}

StringList::Iterator StringList::end(void) const
{
    return (vector != nullptr) ? StringList::Iterator(vector->end()) : StringList::Iterator();
}

StringList::Iterator::Iterator(StringVectorIterator it) : it(it)
{
}

bool StringList::Iterator::operator!=(Iterator rhs)
{
    return it != rhs.it;
}

const char *StringList::Iterator::operator*()
{
    auto v = *it;
    return v->c_str();
}

void StringList::Iterator::operator++()
{
    ++it;
}






Int32List::Int32List(const Int32Vector *vector) : vector(vector)
{
}

Int32List::Int32List(const Int32List& other)
{
    this->vector = other.vector;
}

Int32List& Int32List::operator=(const Int32List& other)
{
    this->vector = other.vector;
    return *this;
}

uint32_t Int32List::size(void) const
{
    return (vector != nullptr) ? vector->size() : 0;
}

int32_t Int32List::operator[](unsigned index) const
{
    return (vector != nullptr) ? vector->Get(index) : 0;
}

Int32List::Iterator Int32List::begin(void) const
{
    return (vector != nullptr) ? Int32List::Iterator(vector->begin()) : Int32List::Iterator();
}

Int32List::Iterator Int32List::end(void) const
{
    return (vector != nullptr) ? Int32List::Iterator(vector->end()) : Int32List::Iterator();
}

Int32List::Iterator::Iterator(Int32VectorIterator it) : it(it)
{
}

bool Int32List::Iterator::operator!=(Iterator rhs)
{
    return it != rhs.it;
}

int32_t Int32List::Iterator::operator*()
{
    return *it;
}

void Int32List::Iterator::operator++()
{
    ++it;
}



FloatList::FloatList(const FloatVector *vector) : vector(vector)
{
}

FloatList::FloatList(const FloatList& other)
{
    this->vector = other.vector;
}

FloatList& FloatList::operator=(const FloatList& other)
{
    this->vector = other.vector;
    return *this;
}

uint32_t FloatList::size(void) const
{
    return (vector != nullptr) ? vector->size() : 0;
}

float FloatList::operator[](unsigned index) const
{
    return (vector != nullptr) ? vector->Get(index) : 0;
}

FloatList::Iterator FloatList::begin(void) const
{
    return (vector != nullptr) ? FloatList::Iterator(vector->begin()) : FloatList::Iterator();
}

FloatList::Iterator FloatList::end(void) const
{
    return (vector != nullptr) ? FloatList::Iterator(vector->end()) : FloatList::Iterator();
}

FloatList::Iterator::Iterator(FloatVectorIterator it) : it(it)
{
}

bool FloatList::Iterator::operator!=(Iterator rhs)
{
    return it != rhs.it;
}

float FloatList::Iterator::operator*()
{
    return *it;
}

void FloatList::Iterator::operator++()
{
    ++it;
}


} // namespace npu_toolkit
