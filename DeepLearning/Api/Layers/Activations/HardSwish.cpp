#include "DeepLearning/Api/Layers/Activations/HardSwish.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("hard_swish", &Thor::HardSwish::deserialize);
    return true;
}();
}  // namespace
