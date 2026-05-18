#include "tflite_micro_model_config.h"
#pragma once

#include <cstdint>
#include <cstring>
#include "ml/third_party/tflm/common.h"
#include "ml/tflite_micro_model/tflite_micro_logger.hpp"


namespace npu_toolkit
{

struct TfliteTensorShape
{
    // The maximum number of dimensions this tensor object can hold
    // Note that 5 is the number that TF uses as well.
    static constexpr const unsigned MAX_DIMENSIONS = 5;

    uint32_t dims[MAX_DIMENSIONS] = { 0 };
    uint8_t length = 0;

    TfliteTensorShape() = default;
    TfliteTensorShape(const TfLiteIntArray* dims);
    void init(const TfLiteIntArray* dims);

    uint32_t flat_size() const;
    uint32_t operator [](int i) const;
    char* to_str(char* str_buffer) const;
};


struct TfliteTensorView : public TfLiteTensor
{
    unsigned element_size() const;
    const char* to_str(char* str_buffer = nullptr) const;
    TfliteTensorShape shape() const;

    template <typename qtype>
    float dequantized_value(int index) const
    {
        const qtype quant_val = static_cast<const qtype*>(data.raw_const)[index];
        return (((float)quant_val) - params.zero_point) * params.scale;
    }

    static TfliteTensorView from_eval_tensor(const TfLiteEvalTensor& eval_tensor)
    {
        TfliteTensorView view;
        memset((void*)&view, 0, sizeof(view));
        view.data = eval_tensor.data;
        view.dims = eval_tensor.dims;
        view.type = eval_tensor.type;
        view.bytes = view.shape().flat_size() * view.element_size();
        return view;
    }

};

template <typename qtype>
float dequantized_value(const TfLiteQuantizationParams& params, qtype value)
{
    return (((float)value) - params.zero_point) * params.scale;
}



const char* to_str(TfLiteType dtype);



} // namespace npu_toolkit