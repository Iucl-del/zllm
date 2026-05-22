#pragma once

#include "tensor.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

/**
 * @brief 计算图容器，负责 Tensor 节点的所有权、输出标记和执行顺序构建。
 *
 * ComputeGraph 只管理 Tensor 元数据对象的生命周期；Tensor::data 指向的真实
 * 权重/激活内存由外部内存资源管理。图中的依赖关系由 Tensor::src 数组表示。
 */
class ComputeGraph {
public:
    /**
     * @brief 创建一个空计算图。
     */
    ComputeGraph() = default;

    /**
     * @brief 析构计算图并释放所有已注册的 Tensor 元数据对象。
     */
    ~ComputeGraph();

    /**
     * @brief 禁止拷贝构造，避免多个图重复释放同一批 Tensor。
     */
    ComputeGraph(const ComputeGraph&) = delete;

    /**
     * @brief 禁止拷贝赋值，避免 Tensor 所有权不清晰。
     */
    ComputeGraph& operator=(const ComputeGraph&) = delete;

    /**
     * @brief 移动构造，将另一个图中的 Tensor 所有权和调度信息转移到当前图。
     * @param other 被移动的计算图。
     */
    ComputeGraph(ComputeGraph&& other) noexcept;

    /**
     * @brief 移动赋值，先释放当前图内容，再接管另一个图的资源。
     * @param other 被移动的计算图。
     * @return 当前计算图引用。
     */
    ComputeGraph& operator=(ComputeGraph&& other) noexcept;

    /**
     * @brief 新建并注册一个通用 Tensor 节点。
     * @tparam N 形状数组的维度数量，最多写入 TENSOR_MAX_DIMS 个维度。
     * @param name Tensor 名称，用于调试、导出和查找。
     * @param dims Tensor 形状，未提供的高维度自动补 0。
     * @param dtype Tensor 数据类型。
     * @param type Tensor 类型，例如输入、权重、激活或输出。
     * @param layer_id Tensor 所属网络层，-1 表示全局节点。
     * @return 新建 Tensor 的裸指针；所有权归 ComputeGraph。
     */
    template <std::size_t N>
    Tensor* add_tensor(const std::string& name,
                       const std::array<int64_t, N>& dims,
                       DataType dtype,
                       TensorType type = TensorType::TENSOR_TYPE_ACTIVATION,
                       int layer_id = -1) {
        Tensor* tensor = new Tensor();
        tensor->name = name;
        tensor->dims = normalize_dims(dims);
        tensor->dtype = dtype;
        tensor->type = type;
        tensor->layer_id = layer_id;
        add_tensor(tensor);
        return tensor;
    }

    /**
     * @brief 新建并注册一个输入 Tensor。
     * @tparam N 形状数组的维度数量。
     * @param name 输入名称。
     * @param dims 输入形状。
     * @param dtype 输入数据类型。
     * @return 新建输入 Tensor 的裸指针；所有权归 ComputeGraph。
     */
    template <std::size_t N>
    Tensor* add_input(const std::string& name,
                      const std::array<int64_t, N>& dims,
                      DataType dtype) {
        return add_tensor(name, dims, dtype, TensorType::TENSOR_TYPE_INPUT);
    }

    /**
     * @brief 新建并注册一个权重 Tensor。
     * @tparam N 形状数组的维度数量。
     * @param name 权重名称，需要和模型文件中的权重名保持一致。
     * @param dims 权重形状。
     * @param dtype 权重数据类型。
     * @param layer_id 权重所属层，-1 表示全局权重。
     * @return 新建权重 Tensor 的裸指针；所有权归 ComputeGraph。
     */
    template <std::size_t N>
    Tensor* add_weight(const std::string& name,
                       const std::array<int64_t, N>& dims,
                       DataType dtype,
                       int layer_id = -1) {
        return add_tensor(name, dims, dtype, TensorType::TENSOR_TYPE_WEIGHT, layer_id);
    }

    /**
     * @brief 新建并注册一个算子输出 Tensor。
     * @tparam SrcN 源 Tensor 数组长度。
     * @tparam DimN 形状数组长度。
     * @param name 算子输出名称。
     * @param op_type 产生该 Tensor 的算子类型。
     * @param srcs 算子输入 Tensor 指针数组。
     * @param dims 输出 Tensor 形状。
     * @param dtype 输出 Tensor 数据类型。
     * @param layer_id 算子所属层，-1 表示全局算子。
     * @return 新建算子输出 Tensor 的裸指针；所有权归 ComputeGraph。
     */
    template <std::size_t SrcN, std::size_t DimN>
    Tensor* add_op(const std::string& name,
                   OperationType op_type,
                   const std::array<Tensor*, SrcN>& srcs,
                   const std::array<int64_t, DimN>& dims,
                   DataType dtype,
                   int layer_id = -1) {
        Tensor* tensor = add_tensor(name, dims, dtype, TensorType::TENSOR_TYPE_ACTIVATION, layer_id);
        tensor->op_type = op_type;
        constexpr std::size_t copy_count = SrcN < TENSOR_MAX_SRC ? SrcN : TENSOR_MAX_SRC;
        for (std::size_t i = 0; i < copy_count; ++i) {
            tensor->src[i] = srcs[i];
        }
        return tensor;
    }

    /**
     * @brief 从一组外部输出反向收集依赖并重建整个图。
     * @param outputs 作为图输出的 Tensor 指针列表。
     *
     * 该接口用于兼容“先手工创建 Tensor，再交给图接管”的构建方式。调用前会清空
     * 当前图并释放旧节点，因此 outputs 必须指向新的、尚未由当前图管理的 Tensor。
     */
    void build_from_outputs(std::initializer_list<Tensor*> outputs);

    /**
     * @brief 释放所有 Tensor 并清空输出、执行顺序、层分组等派生状态。
     */
    void clear();

    /**
     * @brief 基于已注册 Tensor 重新生成拓扑执行顺序和层分组。
     */
    void rebuild_order();

    /**
     * @brief rebuild_order 的语义化别名，用于模型构图代码。
     */
    void rebuild_execution_order();

    /**
     * @brief 将指定 Tensor 标记为外部输出。
     * @param tensor 输出 Tensor。
     */
    void mark_output(Tensor* tensor);

    /**
     * @brief 将外部输出列表中的旧输出替换成新输出。
     * @param old_t 原输出 Tensor。
     * @param new_t 新输出 Tensor。
     */
    void replace_output(Tensor* old_t, Tensor* new_t);

    /**
     * @brief 注册一个已创建的 Tensor，使 ComputeGraph 接管其生命周期。
     * @param tensor 待接管的 Tensor，允许传入 nullptr 且会被忽略。
     */
    void add_tensor(Tensor* tensor);

    /**
     * @brief 在图中插入一个表示跨设备拷贝的 Tensor 节点。
     * @param original 拷贝源 Tensor。
     * @param dst_dev 目标逻辑设备。
     * @return 新建 memcpy Tensor；所有权归 ComputeGraph。
     */
    Tensor* insert_memcpy(Tensor* original, Device dst_dev);

    /**
     * @brief 将计算图导出为 Graphviz DOT 文件。
     * @param path DOT 文件输出路径。
     */
    void export_dot(const std::string& path) const;

    /**
     * @brief 打印计算图摘要，便于快速确认节点数量、输出和层范围。
     */
    void print_summary() const;

    /**
     * @brief 获取拓扑排序后的可执行节点列表。
     * @return 只包含需要计算的 Tensor；输入和权重不会出现在该列表中。
     */
    [[nodiscard]] const std::vector<Tensor*>& get_execution_order() const { return execution_order_; }

    /**
     * @brief 获取按拓扑层级分组的节点列表。
     * @return 每个内部 vector 表示同一依赖层级，层内节点可并行调度。
     */
    [[nodiscard]] const std::vector<std::vector<Tensor*>>& get_execution_levels() const { return execution_levels_; }

    /**
     * @brief 获取图中所有已注册 Tensor。
     * @return Tensor 指针列表；所有权仍归 ComputeGraph。
     */
    [[nodiscard]] const std::vector<Tensor*>& get_all_tensors() const { return all_tensors_; }

    /**
     * @brief 获取被标记为模型外部输出的 Tensor。
     * @return 输出 Tensor 指针列表。
     */
    [[nodiscard]] const std::vector<Tensor*>& get_external_outputs() const { return external_outputs_; }

    /**
     * @brief 获取按 layer_id 聚合的可执行节点。
     * @return layer_id 到 Tensor 列表的映射。
     */
    [[nodiscard]] const std::map<int, std::vector<Tensor*>>& get_layer_groups() const { return layer_groups_; }

    /**
     * @brief 获取图中出现过的最大 layer_id。
     * @return 最大层号；空图或无分层节点时返回 -1。
     */
    [[nodiscard]] int get_max_layer() const { return max_layer_; }

private:
    /**
     * @brief 将任意长度的形状数组规范化为 Tensor 内部固定长度形状。
     * @tparam N 输入形状数组长度。
     * @param dims 输入形状。
     * @return 长度为 TENSOR_MAX_DIMS 的形状数组。
     */
    template <std::size_t N>
    static std::array<int64_t, TENSOR_MAX_DIMS> normalize_dims(const std::array<int64_t, N>& dims) {
        std::array<int64_t, TENSOR_MAX_DIMS> normalized{0, 0, 0, 0};
        constexpr std::size_t copy_count = N < TENSOR_MAX_DIMS ? N : TENSOR_MAX_DIMS;
        for (std::size_t i = 0; i < copy_count; ++i) {
            normalized[i] = dims[i];
        }
        return normalized;
    }

    /**
     * @brief 从输出节点反向广度优先遍历，收集所有依赖节点。
     * @param seeds 输出种子节点。
     */
    void reverse_bfs_collect(const std::vector<Tensor*>& seeds);

    /**
     * @brief 对 all_tensors_ 做拓扑排序，并生成 execution_order_ 和 execution_levels_。
     */
    void topological_sort();

    /**
     * @brief 为 DOT 导出生成稳定节点 ID。
     * @param tensor Tensor 指针。
     * @return DOT 节点 ID 字符串。
     */
    static std::string dot_id(const Tensor* tensor);

    /**
     * @brief 判断某个 Tensor 是否已经由当前图管理。
     * @param tensor 待检查 Tensor。
     * @return 已注册返回 true，否则返回 false。
     */
    [[nodiscard]] bool owns_tensor(const Tensor* tensor) const;

private:
    int max_layer_ = -1;
    std::vector<Tensor*> all_tensors_;
    std::vector<Tensor*> execution_order_;
    std::vector<Tensor*> external_outputs_;
    std::map<int, std::vector<Tensor*>> layer_groups_;
    std::vector<std::vector<Tensor*>> execution_levels_;
};
