#include "gguf_parser.hpp"
#include "model.hpp"

#include <exception>
#include <filesystem>

int main() {
    try {
        auto model_path = std::filesystem::path("resources/Qwen3-0.6B-BF16.gguf");
        if (!std::filesystem::exists(model_path)) {
            model_path = "../resources/Qwen3-0.6B-BF16.gguf";
        }

        GGUFParser ggufParser(model_path);
        auto model = ModelFactory::CreateFromGGUF(ggufParser.get_info());
        if (model) {
            auto graph = model->build_graph(ggufParser.get_info());
            graph->print_summary();
        }
    } catch (const std::exception& error) {
        std::println("error: {}", error.what());
    }



    return 0;
}
