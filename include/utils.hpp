#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

enum class DataType:uint8_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_COUNT   = 40
};

enum class Device:uint8_t {
    CPU,
    CUDA,
    SYCL,
    VULKAN,
    UNKNOWN
};

enum class TensorType:uint8_t {
    TENSOR_TYPE_UNKNOWN,
    TENSOR_TYPE_INPUT,
    TENSOR_TYPE_WEIGHT,
    TENSOR_TYPE_ACTIVATION,
    TENSOR_TYPE_OUTPUT,
    TENSOR_TYPE_CACHE,
    TENSOR_TYPE_VIEW
};

enum class OperationType:uint8_t {
    OP_TYPE_NONE,
    OP_TYPE_EMBEDDING,
    OP_TYPE_RMS_NORM,
    OP_TYPE_MATMUL,
    OP_TYPE_ADD,
    OP_TYPE_MUL,
    OP_TYPE_ROPE,
    OP_TYPE_ATTENTION,
    OP_TYPE_SILU,
    OP_TYPE_OUTPUT,
    OP_TYPE_MEMCPY
};

enum gguf_metadata_value_type : uint32_t {
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

enum class ModelType:uint8_t {
    UNKNOWN,
    CAUSAL_LM,      // 因果语言模型
    EMBEDDING,      // 嵌入模型
    SEQ2SEQ,        // 序列到序列
    CLASSIFIER,     // 分类器
};

enum class ModelArch:uint8_t{
    UNKNOWN,
    QWEN3,
    QWEN35
};

inline std::string data_type_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::GGML_TYPE_F32:     return "F32";
        case DataType::GGML_TYPE_F16:     return "F16";
        case DataType::GGML_TYPE_Q4_0:    return "Q4_0";
        case DataType::GGML_TYPE_Q4_1:    return "Q4_1";
        case DataType::GGML_TYPE_Q5_0:    return "Q5_0";
        case DataType::GGML_TYPE_Q5_1:    return "Q5_1";
        case DataType::GGML_TYPE_Q8_0:    return "Q8_0";
        case DataType::GGML_TYPE_Q8_1:    return "Q8_1";
        case DataType::GGML_TYPE_Q2_K:    return "Q2_K";
        case DataType::GGML_TYPE_Q3_K:    return "Q3_K";
        case DataType::GGML_TYPE_Q4_K:    return "Q4_K";
        case DataType::GGML_TYPE_Q5_K:    return "Q5_K";
        case DataType::GGML_TYPE_Q6_K:    return "Q6_K";
        case DataType::GGML_TYPE_Q8_K:    return "Q8_K";
        case DataType::GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case DataType::GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case DataType::GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case DataType::GGML_TYPE_IQ1_S:   return "IQ1_S";
        case DataType::GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case DataType::GGML_TYPE_IQ3_S:   return "IQ3_S";
        case DataType::GGML_TYPE_IQ2_S:   return "IQ2_S";
        case DataType::GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case DataType::GGML_TYPE_I8:      return "I8";
        case DataType::GGML_TYPE_I16:     return "I16";
        case DataType::GGML_TYPE_I32:     return "I32";
        case DataType::GGML_TYPE_I64:     return "I64";
        case DataType::GGML_TYPE_F64:     return "F64";
        case DataType::GGML_TYPE_IQ1_M:   return "IQ1_M";
        case DataType::GGML_TYPE_BF16:    return "BF16";
        case DataType::GGML_TYPE_TQ1_0:   return "TQ1_0";
        case DataType::GGML_TYPE_TQ2_0:   return "TQ2_0";
        case DataType::GGML_TYPE_MXFP4:   return "MXFP4";
        case DataType::GGML_TYPE_COUNT:   return "COUNT";
        default:                          return "Unknown";
    }
}

inline size_t data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::GGML_TYPE_F32: return 4;
        case DataType::GGML_TYPE_F16: return 2;
        case DataType::GGML_TYPE_BF16: return 2;
        case DataType::GGML_TYPE_I8: return 1;
        case DataType::GGML_TYPE_I16: return 2;
        case DataType::GGML_TYPE_I32: return 4;
        case DataType::GGML_TYPE_I64: return 8;
        case DataType::GGML_TYPE_F64: return 8;
        default: return 0;
    }
}

inline std::string operation_type_to_string(OperationType op) {
    switch (op) {
        case OperationType::OP_TYPE_NONE: return "NONE";
        case OperationType::OP_TYPE_EMBEDDING: return "EMBEDDING";
        case OperationType::OP_TYPE_RMS_NORM: return "RMS_NORM";
        case OperationType::OP_TYPE_MATMUL: return "MATMUL";
        case OperationType::OP_TYPE_ADD: return "ADD";
        case OperationType::OP_TYPE_MUL: return "MUL";
        case OperationType::OP_TYPE_ROPE: return "ROPE";
        case OperationType::OP_TYPE_ATTENTION: return "ATTENTION";
        case OperationType::OP_TYPE_SILU: return "SILU";
        case OperationType::OP_TYPE_OUTPUT: return "OUTPUT";
        default: return "UNKNOWN";
    }
}
