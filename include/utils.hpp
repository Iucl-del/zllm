#pragma once

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