#include "qwen3.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <stdexcept>

namespace {
std::array<int64_t, 2> tensor_dims(const Tensors_info& info) {
    std::array<int64_t, 2> dims{0, 0};
    for (size_t i = 0; i < info.dimensions.size() && i < dims.size(); ++i) {
        dims[i] = static_cast<int64_t>(info.dimensions[i]);
    }
    return dims;
}

Tensor* add_required_weight(ComputeGraph& graph, const gguf_info& info, const std::string& name) {
    const auto it = std::ranges::find_if(info.tensors_info, [&](const Tensors_info& tensor) {
        return tensor.name == name;
    });
    if (it == info.tensors_info.end()) {
        throw std::runtime_error("missing required tensor: " + name);
    }
    const auto dims = tensor_dims(*it);
    return graph.add_weight(name, dims, it->dtype);
}

Tensor* add_required_weight_any(ComputeGraph& graph,
                                const gguf_info& info,
                                std::initializer_list<std::string_view> names) {
    for (std::string_view name : names) {
        const auto it = std::ranges::find_if(info.tensors_info, [&](const Tensors_info& tensor) {
            return tensor.name == name;
        });
        if (it != info.tensors_info.end()) {
            const auto dims = tensor_dims(*it);
            return graph.add_weight(it->name, dims, it->dtype);
        }
    }

    std::string message = "missing required tensor, candidates:";
    for (std::string_view name : names) {
        message += " ";
        message += name;
    }
    throw std::runtime_error(message);
}
}

Qwen3::Qwen3(){
    name_ = "Qwen3";
    model_type_ = ModelType::CAUSAL_LM;
    model_arch_ = ModelArch::QWEN3;
}

Qwen3::~Qwen3() = default;

void Qwen3::print_info()
{
    std::println("{}: hidden={}, layers={}, heads={}/kv={}, head_dim={}, vocab={}, max_seq_len={}",
                 name_,
                 config_.hidden_size,
                 config_.num_layers,
                 config_.num_heads,
                 config_.num_kv_heads,
                 config_.head_dim,
                 config_.vocab_size,
                 config_.max_seq_len);
}

void Qwen3::parse_config(const gguf_info& info)
{
    auto& meta_data = info.meta_data;
    config_.hidden_size       = meta_data["qwen3.embedding_length"].asInt();
    config_.num_layers        = meta_data["qwen3.block_count"].asInt();
    config_.num_heads         = meta_data["qwen3.attention.head_count"].asInt();
    config_.num_kv_heads      = meta_data["qwen3.attention.head_count_kv"].asInt();
    config_.head_dim          = meta_data["qwen3.attention.key_length"].asInt();
    config_.intermediate_size = meta_data["qwen3.feed_forward_length"].asInt();
    config_.rms_norm_eps      = meta_data["qwen3.attention.layer_norm_rms_epsilon"].asFloat();
    config_.rope_theta        = meta_data["qwen3.rope.freq_base"].asFloat();
    config_.max_seq_len       = meta_data["qwen3.context_length"].asInt();
    if (meta_data.isMember("tokenizer.ggml.tokens") && meta_data["tokenizer.ggml.tokens"].isArray()){
        config_.vocab_size = static_cast<int>(meta_data["tokenizer.ggml.tokens"].size());
    }

    if (config_.vocab_size == 0) {
        for (const auto& t : info.tensors_info) {
            if (t.name == "token_embd.weight" && t.dimensions.size() == 2) {
                config_.vocab_size = static_cast<int>(t.dimensions[1]);
                break;
            }
        }
    }

    if (config_.vocab_size == 0)
    {
        std::println("Cannot determine vocab_size");
        return;
    }
    std::println("Config: hidden={}, heads={}/kv={}, head_dim={}, vocab={}, layers={}",
        config_.hidden_size, config_.num_heads, config_.num_kv_heads,
        config_.head_dim, config_.vocab_size, config_.num_layers);
}

Tensor* Qwen3::build_qwen3_layer(
    ComputeGraph& graph,
    Tensor* input,
    const gguf_info& info,
    int layer_idx,
    int hidden_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    float rms_norm_eps,
    Tensor* rope_cos,
    Tensor* rope_sin)
{
    (void)num_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)intermediate_size;
    (void)rms_norm_eps;
    (void)rope_cos;
    (void)rope_sin;

    const auto prefix = std::format("blk.{}.", layer_idx);
    const std::array<int64_t, 2> hidden_dims{1, hidden_size};

    Tensor* attn_norm_w = add_required_weight(graph, info, prefix + "attn_norm.weight");
    Tensor* q_w = add_required_weight(graph, info, prefix + "attn_q.weight");
    Tensor* k_w = add_required_weight(graph, info, prefix + "attn_k.weight");
    Tensor* v_w = add_required_weight(graph, info, prefix + "attn_v.weight");
    Tensor* o_w = add_required_weight(graph, info, prefix + "attn_output.weight");

    Tensor* ffn_norm_w = add_required_weight(graph, info, prefix + "ffn_norm.weight");
    Tensor* up_w = add_required_weight(graph, info, prefix + "ffn_up.weight");
    Tensor* gate_w = add_required_weight(graph, info, prefix + "ffn_gate.weight");
    Tensor* down_w = add_required_weight(graph, info, prefix + "ffn_down.weight");

    std::array<Tensor*, 2> attn_norm_src{input, attn_norm_w};
    Tensor* attn_norm = graph.add_op(prefix + "attn_norm",
                                      OperationType::OP_TYPE_RMS_NORM,
                                      attn_norm_src,
                                      hidden_dims,
                                      input->dtype,
                                      layer_idx);

    std::array<Tensor*, 2> q_src{attn_norm, q_w};
    Tensor* q = graph.add_op(prefix + "q", OperationType::OP_TYPE_MATMUL, q_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 2> k_src{attn_norm, k_w};
    Tensor* k = graph.add_op(prefix + "k", OperationType::OP_TYPE_MATMUL, k_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 2> v_src{attn_norm, v_w};
    Tensor* v = graph.add_op(prefix + "v", OperationType::OP_TYPE_MATMUL, v_src, hidden_dims, input->dtype, layer_idx);

    std::array<Tensor*, 3> attn_src{q, k, v};
    Tensor* attn = graph.add_op(prefix + "attention",
                                 OperationType::OP_TYPE_ATTENTION,
                                 attn_src,
                                 hidden_dims,
                                 input->dtype,
                                 layer_idx);

    std::array<Tensor*, 2> attn_out_src{attn, o_w};
    Tensor* attn_out = graph.add_op(prefix + "attn_output",
                                     OperationType::OP_TYPE_MATMUL,
                                     attn_out_src,
                                     hidden_dims,
                                     input->dtype,
                                     layer_idx);

    std::array<Tensor*, 2> residual_src{input, attn_out};
    Tensor* residual = graph.add_op(prefix + "attn_residual",
                                     OperationType::OP_TYPE_ADD,
                                     residual_src,
                                     hidden_dims,
                                     input->dtype,
                                     layer_idx);

    std::array<Tensor*, 2> ffn_norm_src{residual, ffn_norm_w};
    Tensor* ffn_norm = graph.add_op(prefix + "ffn_norm",
                                     OperationType::OP_TYPE_RMS_NORM,
                                     ffn_norm_src,
                                     hidden_dims,
                                     input->dtype,
                                     layer_idx);

    std::array<Tensor*, 2> gate_src{ffn_norm, gate_w};
    Tensor* gate = graph.add_op(prefix + "ffn_gate", OperationType::OP_TYPE_MATMUL, gate_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 1> silu_src{gate};
    Tensor* silu = graph.add_op(prefix + "ffn_silu", OperationType::OP_TYPE_SILU, silu_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 2> up_src{ffn_norm, up_w};
    Tensor* up = graph.add_op(prefix + "ffn_up", OperationType::OP_TYPE_MATMUL, up_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 2> gated_src{silu, up};
    Tensor* gated = graph.add_op(prefix + "ffn_gated", OperationType::OP_TYPE_MUL, gated_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 2> down_src{gated, down_w};
    Tensor* down = graph.add_op(prefix + "ffn_down", OperationType::OP_TYPE_MATMUL, down_src, hidden_dims, input->dtype, layer_idx);
    std::array<Tensor*, 2> output_src{residual, down};
    return graph.add_op(prefix + "output", OperationType::OP_TYPE_ADD, output_src, hidden_dims, input->dtype, layer_idx);
}

std::unique_ptr<ComputeGraph> Qwen3::build_graph(const gguf_info& info)
{
    parse_config(info);

    auto graph = std::make_unique<ComputeGraph>();
    const std::array<int64_t, 2> hidden_dims{1, config_.hidden_size};
    const std::array<int64_t, 2> token_dims{1, 1};

    Tensor* input = graph->add_input("input_ids", token_dims, DataType::GGML_TYPE_I32);

    Tensor* token_embd = add_required_weight(*graph, info, "token_embd.weight");
    std::array<Tensor*, 2> embd_src{input, token_embd};
    Tensor* current = graph->add_op("token_embedding",
                                    OperationType::OP_TYPE_EMBEDDING,
                                    embd_src,
                                    hidden_dims,
                                    token_embd->dtype);

    Tensor* rope_cos = graph->add_tensor("rope_cos",
                                         hidden_dims,
                                         DataType::GGML_TYPE_F32,
                                         TensorType::TENSOR_TYPE_ACTIVATION);
    Tensor* rope_sin = graph->add_tensor("rope_sin",
                                         hidden_dims,
                                         DataType::GGML_TYPE_F32,
                                         TensorType::TENSOR_TYPE_ACTIVATION);

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        current = build_qwen3_layer(*graph,
                                    current,
                                    info,
                                    layer,
                                    config_.hidden_size,
                                    config_.num_heads,
                                    config_.num_kv_heads,
                                    config_.head_dim,
                                    config_.intermediate_size,
                                    config_.rms_norm_eps,
                                    rope_cos,
                                    rope_sin);
    }

    Tensor* output_norm_w = add_required_weight(*graph, info, "output_norm.weight");
    std::array<Tensor*, 2> norm_src{current, output_norm_w};
    Tensor* norm = graph->add_op("output_norm",
                                 OperationType::OP_TYPE_RMS_NORM,
                                 norm_src,
                                 hidden_dims,
                                 current->dtype);

    Tensor* output_w = add_required_weight_any(*graph, info, {"output.weight", "token_embd.weight"});
    const std::array<int64_t, 2> logits_dims{1, config_.vocab_size};
    std::array<Tensor*, 2> logits_src{norm, output_w};
    Tensor* logits = graph->add_op("logits",
                                   OperationType::OP_TYPE_MATMUL,
                                   logits_src,
                                   logits_dims,
                                   norm->dtype);
    graph->mark_output(logits);
    graph->rebuild_execution_order();
    return graph;
}
