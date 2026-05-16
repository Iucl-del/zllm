#include "gguf_parser.hpp"

int main() {
    GGUFParser ggufParser("../resources/Qwen3-0.6B-BF16.gguf");
    ggufParser.get_info().print_info();
    return 0;
}
