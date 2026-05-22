#include "gguf_parser.hpp"
#include "global.hpp"
#include "tensor.hpp"

template<typename T>
requires std::is_trivially_copyable_v<T>
bool read_binary(std::fstream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

bool read_binary(std::fstream& in, void* data, std::streamsize size) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data), size));
}

uint64_t align_to(uint64_t offset, uint64_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

GGUFParser::GGUFParser(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        std::println("gguf file does not exist, path: {}", path.string());
        return;
    }

    file_.open(path, std::ios::in | std::ios::binary);
    if (!file_.is_open()) {
        std::println("failed to open gguf file, path: {}", path.string());
        return;
    }

    char name[4] {};
    if (!file_.read(name, 4) || std::strncmp(name, "GGUF", 4) != 0) {
        std::println("invalid gguf file: magic mismatch");
        return;
    }

    parser();
}

GGUFParser::~GGUFParser() {
    file_.close();
}

void GGUFParser::parser() {
    if (!read_binary(file_, info_.version) ||
        !read_binary(file_, info_.tensor_count) ||
        !read_binary(file_, info_.kv_count)) {
        std::println("failed to read gguf header");
        return;
    }

    if (info_.version != 3) {
        std::println("unsupported gguf version: {}", info_.version);
        return;
    }
    if (!info_.kv_count) {
        std::println("kv_count is zero");
        return;
    }

    read_meta_data();
    read_tensors_info();
    uint64_t alignment = 32; // GGUF 默认一般是 32
    data_offset_ = align_to(static_cast<uint64_t>(file_.tellg()), alignment);

}

void GGUFParser::read_meta_data()
{
    uint64_t len{0};
    std::string name;
    for (size_t i = 0; i < info_.kv_count; ++i) {
        if (!read_binary(file_, len)) {
            return;
        }

        name.resize(len, 0);
        if (!read_binary(file_, name.data(), static_cast<std::streamsize>(len))) {
            return;
        }

        gguf_metadata_value_type type;
        if (!read_binary(file_, type)) {
            return;
        }

        Json::Value value;
        if (!read_metadata_value(type, value)) {
            std::println("metadata value read failed: {}", name);
            return;
        }

        info_.meta_data[name] = value;
        len = 0;
        name.clear();
    }
}

bool GGUFParser::read_metadata_value(gguf_metadata_value_type type, Json::Value& value)
{
    switch (type)
    {
        case GGUF_METADATA_VALUE_TYPE_UINT8: {
            uint8_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::UInt(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_INT8: {
            int8_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::Int(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_UINT16: {
            uint16_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::UInt(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_INT16: {
            int16_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::Int(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_UINT32: {
            uint32_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::UInt(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_INT32: {
            int32_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::Int(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
            float v;
            if (!read_binary(file_, v)) return false;
            value = v;
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_BOOL: {
            bool v;
            if (!read_binary(file_, v)) return false;
            value = v;
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_STRING: {
            uint64_t len;
            if (!read_binary(file_, len)) return false;

            std::string str;
            str.resize(len, 0);
            if (!read_binary(file_, str.data(), static_cast<std::streamsize>(len))) return false;

            value = str;
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_ARRAY: {
            gguf_metadata_value_type item_type;
            uint64_t len;
            if (!read_binary(file_, item_type) || !read_binary(file_, len)) return false;

            value = Json::arrayValue;
            for (uint64_t i = 0; i < len; ++i) {
                Json::Value item;
                if (!read_metadata_value(item_type, item)) return false;
                value.append(item);
            }
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64: {
            uint64_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::UInt64(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_INT64: {
            int64_t v;
            if (!read_binary(file_, v)) return false;
            value = Json::Int64(v);
            return true;
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
            double v;
            if (!read_binary(file_, v)) return false;
            value = v;
            return true;
        }
        default:
            return false;
    }
}

void GGUFParser::read_tensors_info()
{
    if (!info_.tensor_count) {
        std::println("tensor_count is zero");
        return;
    }

    uint64_t len{0};
    for (size_t i = 0; i < info_.tensor_count; ++i) {
        Tensors_info tensors_info;

        if (!read_binary(file_, len)) {
            return;
        }

        tensors_info.name.resize(len, 0);
        if (!read_binary(file_, tensors_info.name.data(), static_cast<std::streamsize>(len))) {
            return;
        }

        if (!read_binary(file_, tensors_info.dim_num)) {
            return;
        }
        if (tensors_info.dim_num < 1 || tensors_info.dim_num > TENSOR_MAX_DIMS) {
            std::println("unsupported tensor dim count: {}", tensors_info.dim_num);
            return;
        }

        tensors_info.dimensions.resize(tensors_info.dim_num, 0);
        for (size_t n = 0; n < tensors_info.dim_num; ++n) {
            if (!read_binary(file_, tensors_info.dimensions[tensors_info.dim_num - n - 1])) {
                return;
            }
        }
        uint32_t dtype{};
        if (!read_binary(file_, dtype)) {
            return;
        }
        tensors_info.dtype = static_cast<DataType>(dtype);
        if (!read_binary(file_, tensors_info.off_set)) {
            return;
        }

        info_.tensors_info.emplace_back(std::move(tensors_info));
        len = 0;
    }
}

void gguf_info::print_info()
{
    auto get_meta_string = [this](const std::string_view key) -> std::string
    {
        if (meta_data.isMember(key.data()) && meta_data[key.data()].isString()) {
            return meta_data[key.data()].asString();
        }
        return "unknown";
    };
    std::println("{:-<14}GGUF 文件信息{:-<14}", "", "");
    std::println("gguf version:             {}", version);
    std::println("model arch:               {}", get_meta_string("general.architecture"));
    std::println("model name:               {}", get_meta_string("general.name"));
    std::println("kv_count:                 {}", kv_count);
    std::println("tensor_count:             {}", tensor_count);
    std::println("data_offset:              {}", off_set);
    std::println("{:-<26} {:-<8} {:-<14}", "", "", "");
    std::println("{:<26} {:<8} {}", "名称", "数据类型", "维度");
    std::println("{:-<26} {:-<8} {:-<14}", "", "", "");
    for (const auto& info : tensors_info) {
        // std::println("(\"{}\",\"{}\",{}),", info.name, data_type_to_string(info.dtype), info.dimensions);
        std::println("{:<26} {:<8} {}", info.name, data_type_to_string(info.dtype), info.dimensions);
    }
    std::println("{:-<26} {:-<8} {:-<14}", "", "", "");
}

std::string gguf_info::get_model_architecture()
{
    if (meta_data.isMember("general.architecture") && meta_data["general.architecture"].isString()) {
        return meta_data["general.architecture"].asString();
    }
    return "unknown";
}
