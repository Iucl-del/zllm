#include "graph.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <print>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

/**
 * @brief 将 Tensor 指针按 layer_id 和 name 排序，保证调试输出稳定。
 * @param tensors 待排序 Tensor 指针列表。
 */
void sort_tensors_stably(std::vector<Tensor*>& tensors) {
    std::sort(tensors.begin(), tensors.end(), [](const Tensor* lhs, const Tensor* rhs) {
        if (lhs->layer_id != rhs->layer_id) {
            return lhs->layer_id < rhs->layer_id;
        }
        return lhs->name < rhs->name;
    });
}

/**
 * @brief 将 Tensor 类型映射成 DOT 节点填充色。
 * @param tensor Tensor 指针。
 * @return Graphviz 颜色字符串。
 */
const char* dot_color(const Tensor* tensor) {
    switch (tensor->type) {
        case TensorType::TENSOR_TYPE_INPUT:
            return "#90CAF9";
        case TensorType::TENSOR_TYPE_WEIGHT:
            return "#FFD54F";
        case TensorType::TENSOR_TYPE_OUTPUT:
            return "#EF9A9A";
        case TensorType::TENSOR_TYPE_CACHE:
            return "#B39DDB";
        case TensorType::TENSOR_TYPE_VIEW:
            return "#B0BEC5";
        case TensorType::TENSOR_TYPE_ACTIVATION:
            return "#C8E6C9";
        case TensorType::TENSOR_TYPE_UNKNOWN:
        default:
            return "#ECEFF1";
    }
}

/**
 * @brief 将逻辑设备枚举转换为可读字符串。
 * @param device 逻辑设备。
 * @return 设备名称字符串。
 */
std::string device_to_string(Device device) {
    switch (device) {
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
        case Device::SYCL:
            return "SYCL";
        case Device::VULKAN:
            return "VULKAN";
        case Device::UNKNOWN:
        default:
            return "UNKNOWN";
    }
}

} // namespace

ComputeGraph::~ComputeGraph() {
    clear();
}

ComputeGraph::ComputeGraph(ComputeGraph&& other) noexcept
    : max_layer_(other.max_layer_),
      all_tensors_(std::move(other.all_tensors_)),
      execution_order_(std::move(other.execution_order_)),
      external_outputs_(std::move(other.external_outputs_)),
      layer_groups_(std::move(other.layer_groups_)),
      execution_levels_(std::move(other.execution_levels_)) {
    other.max_layer_ = -1;
}

ComputeGraph& ComputeGraph::operator=(ComputeGraph&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    clear();
    max_layer_ = other.max_layer_;
    all_tensors_ = std::move(other.all_tensors_);
    execution_order_ = std::move(other.execution_order_);
    external_outputs_ = std::move(other.external_outputs_);
    layer_groups_ = std::move(other.layer_groups_);
    execution_levels_ = std::move(other.execution_levels_);
    other.max_layer_ = -1;
    return *this;
}

void ComputeGraph::build_from_outputs(std::initializer_list<Tensor*> outputs) {
    clear();
    external_outputs_.assign(outputs.begin(), outputs.end());
    reverse_bfs_collect(external_outputs_);
    topological_sort();

    for (Tensor* tensor : all_tensors_) {
        tensor->device = Device::UNKNOWN;
    }
}

void ComputeGraph::clear() {
    std::unordered_set<Tensor*> deleted;
    for (Tensor* tensor : all_tensors_) {
        if (tensor && deleted.insert(tensor).second) {
            delete tensor;
        }
    }

    max_layer_ = -1;
    all_tensors_.clear();
    execution_order_.clear();
    external_outputs_.clear();
    layer_groups_.clear();
    execution_levels_.clear();
}

void ComputeGraph::rebuild_order() {
    topological_sort();
}

void ComputeGraph::rebuild_execution_order() {
    rebuild_order();
}

void ComputeGraph::mark_output(Tensor* tensor) {
    if (!tensor) {
        return;
    }

    if (!owns_tensor(tensor)) {
        add_tensor(tensor);
    }

    if (std::find(external_outputs_.begin(), external_outputs_.end(), tensor) == external_outputs_.end()) {
        external_outputs_.push_back(tensor);
    }
    tensor->type = TensorType::TENSOR_TYPE_OUTPUT;
}

void ComputeGraph::replace_output(Tensor* old_t, Tensor* new_t) {
    for (Tensor*& output : external_outputs_) {
        if (output == old_t) {
            output = new_t;
        }
    }

    if (old_t && old_t->type == TensorType::TENSOR_TYPE_OUTPUT) {
        old_t->type = TensorType::TENSOR_TYPE_ACTIVATION;
    }
    if (new_t) {
        if (!owns_tensor(new_t)) {
            add_tensor(new_t);
        }
        new_t->type = TensorType::TENSOR_TYPE_OUTPUT;
    }
}

void ComputeGraph::add_tensor(Tensor* tensor) {
    if (!tensor || owns_tensor(tensor)) {
        return;
    }
    all_tensors_.push_back(tensor);
}

Tensor* ComputeGraph::insert_memcpy(Tensor* original, Device dst_dev) {
    if (!original) {
        throw std::invalid_argument("ComputeGraph::insert_memcpy received nullptr");
    }

    Tensor* proxy = new Tensor();
    proxy->name = "memcpy_" + original->name;
    proxy->op_type = OperationType::OP_TYPE_MEMCPY;
    proxy->type = TensorType::TENSOR_TYPE_ACTIVATION;
    proxy->device = dst_dev;
    proxy->dtype = original->dtype;
    proxy->dims = original->dims;
    proxy->layer_id = original->layer_id;
    proxy->src[0] = original;
    add_tensor(proxy);
    return proxy;
}

void ComputeGraph::export_dot(const std::string& path) const {
    std::ofstream os(path);
    if (!os) {
        throw std::runtime_error(std::format("Cannot open file: {}", path));
    }

    os << "digraph ComputeGraph {\n"
       << "  rankdir=TB;\n"
       << "  node [shape=box, style=filled];\n\n";

    for (const Tensor* tensor : all_tensors_) {
        std::vector<int64_t> real_dims;
        for (int64_t dim : tensor->dims) {
            if (dim != 0) {
                real_dims.push_back(dim);
            }
        }

        const std::string layer = tensor->layer_id >= 0
            ? std::format("L{}", tensor->layer_id)
            : "Global";
        std::string label = std::format("[{},{}] {}", layer, data_type_to_string(tensor->dtype), tensor->name);
        label += std::format("\\n{}{}", device_to_string(tensor->device), real_dims);
        if (tensor->op_type != OperationType::OP_TYPE_NONE) {
            label += "\\n" + operation_type_to_string(tensor->op_type);
        }

        os << std::format("  \"{}\" [fillcolor=\"{}\", label=\"{}\"];\n",
                          dot_id(tensor),
                          dot_color(tensor),
                          label);
    }

    os << '\n';
    for (const Tensor* tensor : all_tensors_) {
        for (Tensor* src : tensor->src) {
            if (!src) {
                continue;
            }
            const char* style = tensor->type == TensorType::TENSOR_TYPE_VIEW ? "style=dashed" : "";
            os << std::format("  \"{}\" -> \"{}\" [{}];\n", dot_id(src), dot_id(tensor), style);
        }
    }

    os << "}\n";
}

void ComputeGraph::print_summary() const {
    std::println("ComputeGraph: tensors={}, executable={}, outputs={}, levels={}, max_layer={}",
                 all_tensors_.size(),
                 execution_order_.size(),
                 external_outputs_.size(),
                 execution_levels_.size(),
                 max_layer_);

    if (!external_outputs_.empty()) {
        std::print("Outputs:");
        for (const Tensor* tensor : external_outputs_) {
            std::print(" {}", tensor ? tensor->name : "<null>");
        }
        std::println("");
    }
}

void ComputeGraph::reverse_bfs_collect(const std::vector<Tensor*>& seeds) {
    std::unordered_set<Tensor*> visited;
    std::queue<Tensor*> queue;

    for (Tensor* tensor : seeds) {
        if (tensor && visited.insert(tensor).second) {
            queue.push(tensor);
        }
    }

    while (!queue.empty()) {
        Tensor* current = queue.front();
        queue.pop();
        all_tensors_.push_back(current);

        for (Tensor* src : current->src) {
            if (src && visited.insert(src).second) {
                queue.push(src);
            }
        }
    }
}

void ComputeGraph::topological_sort() {
    std::unordered_map<Tensor*, int> in_degree;
    std::unordered_multimap<Tensor*, Tensor*> forward_edges;

    for (Tensor* tensor : all_tensors_) {
        in_degree.emplace(tensor, 0);
    }

    for (Tensor* tensor : all_tensors_) {
        for (Tensor* src : tensor->src) {
            if (src && in_degree.contains(src)) {
                ++in_degree[tensor];
                forward_edges.emplace(src, tensor);
            }
        }
    }

    execution_order_.clear();
    execution_levels_.clear();
    layer_groups_.clear();
    max_layer_ = -1;

    std::size_t resolved_count = 0;
    while (resolved_count < all_tensors_.size()) {
        std::vector<Tensor*> level;
        for (Tensor* tensor : all_tensors_) {
            if (in_degree[tensor] == 0) {
                level.push_back(tensor);
            }
        }

        if (level.empty()) {
            break;
        }

        sort_tensors_stably(level);
        for (Tensor* tensor : level) {
            in_degree[tensor] = -1;
            ++resolved_count;

            max_layer_ = std::max(max_layer_, tensor->layer_id);
            if (tensor->is_computed()) {
                execution_order_.push_back(tensor);
                layer_groups_[tensor->layer_id].push_back(tensor);
            }

            auto [begin, end] = forward_edges.equal_range(tensor);
            for (auto it = begin; it != end; ++it) {
                Tensor* next = it->second;
                if (in_degree[next] > 0) {
                    --in_degree[next];
                }
            }
        }

        execution_levels_.push_back(std::move(level));
    }

    if (resolved_count != all_tensors_.size()) {
        throw std::runtime_error(std::format("ComputeGraph: cycle detected, {}/{} resolved",
                                             resolved_count,
                                             all_tensors_.size()));
    }
}

std::string ComputeGraph::dot_id(const Tensor* tensor) {
    if (!tensor) {
        return "null";
    }
    return std::format("L{}_{}", tensor->layer_id, tensor->name);
}

bool ComputeGraph::owns_tensor(const Tensor* tensor) const {
    return std::find(all_tensors_.begin(), all_tensors_.end(), tensor) != all_tensors_.end();
}
