#include "model.hpp"
#include "qwen3.hpp"

Model_Base::~Model_Base() = default;

void ModelFactory::init()
{
    ModelFactory::register_model<Qwen3>("qwen3");
}

namespace
{
    struct ModelFactoryInitializer
    {
        ModelFactoryInitializer()
        {
            ModelFactory::init();
        }
    };
    static ModelFactoryInitializer g_model_factory_init;
}
