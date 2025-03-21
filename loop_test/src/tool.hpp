#ifndef _TOOL_HPP
#define _TOOL_HPP

#include <random>
#include <fstream>
#include <cstdint>
#include "MNN/Tensor.hpp"

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

#define DUMP_NUM_DATA(type)                          \
    auto data = tensor->host<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\t"; \
        }                                            \
        outputOs << "\n";                            \
    }

#define DUMP_CHAR_DATA(type)                                           \
    auto data = tensor->host<type>();                                  \
    for (int z = 0; z < outside; ++z) {                                \
        for (int x = 0; x < width; ++x) {                              \
            outputOs << static_cast<int>(data[x + z * width]) << "\t"; \
        }                                                              \
        outputOs << "\n";                                              \
    }

static void dumpTensor2File(const Tensor* tensor, const char* file) {
    std::ofstream outputOs(file);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    const int outside = tensor->elementSize() / width;

    const auto dataType  = type.code;
    const auto dataBytes = type.bytes();

    if (dataType == halide_type_float) {
        DUMP_NUM_DATA(float);
    }
    if (dataType == halide_type_int && dataBytes == 4) {
        DUMP_NUM_DATA(int32_t);
    }
    if (dataType == halide_type_uint && dataBytes == 1) {
        DUMP_CHAR_DATA(uint8_t);
    }
    if (dataType == halide_type_int && dataBytes == 1) {
        DUMP_CHAR_DATA(int8_t);
    }
}

class Linear {
private:
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    int in_features;
    int out_features;

    void kaiming_uniform_init() {
        // 计算bound
        float bound = sqrt(1.0f / in_features);
        
        // 随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-bound, bound);

        // 初始化权重
        for (int i = 0; i < out_features; i++) {
            for (int j = 0; j < in_features; j++) {
                weights[i][j] = dis(gen);
            }
        }

        // 初始化偏置
        for (int i = 0; i < out_features; i++) {
            bias[i] = dis(gen);
        }
    }

public:
    Linear(int in_features, int out_features) {
        this->in_features = in_features;
        this->out_features = out_features;
        
        weights.resize(out_features, std::vector<float>(in_features));
        bias.resize(out_features);
        
        // 调用初始化
        kaiming_uniform_init();
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output(out_features);
        
        // y = Wx + b
        for (int i = 0; i < out_features; i++) {
            output[i] = bias[i];
            for (int j = 0; j < in_features; j++) {
                output[i] += weights[i][j] * input[j];
            }
        }
        
        return output;
    }
}; 

#endif