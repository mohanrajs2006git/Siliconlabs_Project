#include "tflite_micro_model_config.h"
#pragma once

#include <array>

namespace cpputils
{


template<typename T, unsigned SIZE>
class ArrayHelper : public std::array<T, SIZE>
{
public:

  int count() const
  {
    return _count;
  }

  bool contains(const T& needle) const
  {
    for(int i = 0; i < _count; ++i)
    {
     if(std::array<T, SIZE>::at(i) == needle)
     {
      return true;
     }
    }
    return false;
  }

  bool push_back(const T& v, bool unique=false)
  {
    if(_count == SIZE)
    {
      assert(!"Overflow");
      return false;
    }
    if(unique && contains(v))
    {
      return false;
    }
    this->data()[_count] = v;
    ++_count;
    return true;
  }

  T pop()
  {
    if(_count == 0)
    {
      assert(!"Underflow");
    }
    T last = this[_count-1];
    --_count;
    return last;
  }

  bool remove(const T& v)
  {
    if(_count == 0)
    {
      return false;
    }

    for(int i = 0; i < _count; ++i)
    {
      auto& e = this->data()[i];
      if(memcmp(&e, &v, SIZE) == 0)
      {
        if(i < _count-1)
        {
          const int n = _count - i - 1;
          auto& next = this->data()[i+1];
          memmove(&e, &next, n * SIZE);
        }
        --_count;
        return true;
      }
    }

    return false;
  }


  T* end()
  {
    return &std::array<T, SIZE>::data()[_count];
  }

  const T* end() const
  {
    return &std::array<T, SIZE>::data()[_count];
  }

protected:
  int _count = 0;
};

}