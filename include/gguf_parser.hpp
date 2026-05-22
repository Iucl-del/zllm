#pragma once
#include "global.hpp"
#include <json.h>
#include "utils.hpp"


struct Tensors_info{
    std::string name;
    uint32_t dim_num;
    std::vector<uint64_t> dimensions;
    DataType dtype;
    uint64_t off_set;
    bool transpose = false;
};

struct gguf_info {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    Json::Value meta_data;
    std::vector<Tensors_info> tensors_info;
    uint32_t off_set;

    void print_info();
    std::string get_model_architecture();
};

class GGUFParser {
public:
    explicit GGUFParser(const std::filesystem::path& path);
    ~GGUFParser();
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;
    GGUFParser(GGUFParser&&) = delete;
    GGUFParser& operator=(GGUFParser&&) = delete;
    gguf_info& get_info() { return info_; }
private:
    void parser();
    void read_meta_data();
    bool read_metadata_value(gguf_metadata_value_type type, Json::Value& value);
    void read_tensors_info();
private:
    gguf_info info_;
    std::fstream file_;
    uint64_t data_offset_{0};
};
