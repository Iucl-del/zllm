#pragma once
#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <unordered_map>

#include "global.hpp"
#include "utils.hpp"
#include "gguf_parser.hpp"
#include "graph.hpp"

class Model_Base
{
public:
    virtual ~Model_Base();
    std::string name_;
    virtual void print_info() = 0;
    [[nodiscard]] virtual std::unique_ptr<ComputeGraph> build_graph(const gguf_info& info) = 0;
    [[nodiscard]] virtual size_t vocab_size() const = 0;
    [[nodiscard]] virtual int64_t max_seq_len() const = 0;
protected:
    ModelType model_type_;
    ModelArch model_arch_;
};

class ModelFactory
{
public:
    using ModelCreateFunc = std::function<std::unique_ptr<Model_Base>()>;

    template <typename T>
    static void register_model(const std::string& name)
    {
        model_factories[name] = &create_model<T>;
    }

    [[nodiscard]] static std::unique_ptr<Model_Base> CreateFromArch(const std::string& arch_name) {
        auto it = model_factories.find(arch_name);
        if (it == model_factories.end()) {
            std::println("Unsupported model architecture: {}", arch_name);
            return std::unique_ptr<Model_Base>(nullptr);
        }
        return it->second();
    }

    [[nodiscard]] static std::unique_ptr<Model_Base> CreateFromGGUF(gguf_info& info) {
        std::string arch = info.get_model_architecture(); // qwen3
        // 转换为小写以匹配
        std::transform(arch.begin(), arch.end(), arch.begin(),[](unsigned char c){ return std::tolower(c); });
        return ModelFactory::CreateFromArch(arch);
    }

    static void init();

private:
    template <typename T>
    static std::unique_ptr<Model_Base> create_model()
    {
        return std::make_unique<T>();
    }

private:
    static inline std::unordered_map<std::string, ModelCreateFunc> model_factories;
};
