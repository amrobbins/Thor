#include "DeepLearning/Api/Layers/Activations/HardTanh.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("hard_tanh", &Thor::HardTanh::deserialize);
    return true;
}();
}  // namespace
