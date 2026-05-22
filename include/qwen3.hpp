#pragma once
#include "model.hpp"
#include "gguf_parser.hpp"
#include "tensor.hpp"

// Qwen3 模型结构参数（从 GGUF metadata 解析）
struct Qwen3Config {
    int hidden_size = 1024;//单个token大小
    int num_layers = 28;
    int num_heads = 16;//Q注意力头
    int num_kv_heads = 8;//K/V注意力头
    int head_dim = 128; //d_k
    int intermediate_size = 3072;
    int vocab_size = 0;//token_id 的总数量，也就是 embedding 表有多少行。
    int max_seq_len = 40960;//最大上下文 token 数，不是文字数，是 tokenizer 切分后的 token 数。
    float rms_norm_eps = 1e-6f; //防止除0
    float rope_theta = 1000000.0f;//RoPE 位置下标
};

class Qwen3 : public Model_Base
{
public:
    Qwen3();
    ~Qwen3() override;
    void print_info() override;
    [[nodiscard]] std::unique_ptr<ComputeGraph> build_graph(const gguf_info& info) override;

private:
    void parse_config(const gguf_info& info);
    [[nodiscard]] Tensor* build_qwen3_layer(
        ComputeGraph& graph,
        Tensor* input,           // [batch, seq_len, hidden_size]
        const gguf_info& info,
        int layer_idx,
        int hidden_size,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int intermediate_size,
        float rms_norm_eps,
        Tensor* rope_cos,
        Tensor* rope_sin
    );

    [[nodiscard]] const Qwen3Config& config() const { return config_; }
    [[nodiscard]]size_t vocab_size() const override{ return config_.vocab_size; }

    [[nodiscard]] int64_t max_seq_len() const override { return config_.max_seq_len; }


    Qwen3(const Qwen3&) = delete;
    Qwen3& operator=(const Qwen3&) = delete;
    Qwen3(Qwen3&&) = default;
    Qwen3& operator=(Qwen3&&) = default;
private:
    Qwen3Config config_;
};
