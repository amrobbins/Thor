#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("soft_plus", &Thor::SoftPlus::deserialize);
    return true;
}();
}
