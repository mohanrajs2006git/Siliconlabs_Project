#include "tflite_micro_model_config.h"
#pragma once

#include <cassert>
#include <cstring>
#include <vector>
#include <tuple>

namespace cpputils
{


template<typename T>
class VectorDict : public std::vector<std::tuple<const char*,T>>
{
public:
  bool contains(const char* key) const
  {
    assert(key != nullptr);

    for(const auto& e : *this)
    {
      if(strcmp(std::get<0>(e), key) == 0)
      {
        return true;
      }
    }
    return false;
  }

  T get(const char* key) const
  {
    assert(key != nullptr);

    for(const auto& e : *this)
    {
      if(strcmp(std::get<0>(e), key) == 0)
      {
        return std::get<1>(e);
      }
    }
    assert("Failed to find value");
    for(;;);
  }

  T& get(const char* key)
  {
    assert(key != nullptr);

    for(auto& e : *this)
    {
      if(strcmp(std::get<0>(e), key) == 0)
      {
        return std::get<1>(e);
      }
    }
    assert("Failed to find value");
    for(;;);
  }

  T& get(const char* key, const T& default_value)
  {
    assert(key != nullptr);

    for(auto& e : this)
    {
      if(strcmp(std::get<0>(e), key) == 0)
      {
        return std::get<1>(e);
      }
    }

    put(key, default_value);

    return get(key);
  }

  bool put(
    const char* key,
    const T& value,
    bool unique=true
  )
  {
    assert(key != nullptr);

    for(auto& e : *this)
    {
      if(strcmp(std::get<0>(e), key) == 0)
      {
        if(unique)
        {
          return false;
        }
        else
        {
          std::get<1>(e) = value;
          return true;
        }
      }
    }

    this->push_back({key, value});
    return true;
  }

  T pop(const char* key)
  {
    assert(key != nullptr);

    for(int i = 0; i < this->size(); ++i)
    {
      const auto& e = (*this)[i];
      if(strcmp(std::get<0>(e), key) == 0)
      {
        T retval = std::get<1>(e);
        this->erase(this->begin() + i);
        return retval;
      }
    }

    assert("Failed to find value");
    for(;;);
  }

};

} // namespace cpputils