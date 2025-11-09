#include "DeepLearning/Api/Layers/Activations/Tanh.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("tanh", &Thor::Tanh::deserialize);
    return true;
}();
}
