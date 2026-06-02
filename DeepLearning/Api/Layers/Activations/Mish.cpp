#include "DeepLearning/Api/Layers/Activations/Mish.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("mish", &Thor::Mish::deserialize);
    return true;
}();
}  // namespace
