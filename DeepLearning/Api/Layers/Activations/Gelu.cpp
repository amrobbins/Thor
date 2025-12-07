#include "DeepLearning/Api/Layers/Activations/Gelu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("gelu", &Thor::Gelu::deserialize);
    return true;
}();
}  // namespace
