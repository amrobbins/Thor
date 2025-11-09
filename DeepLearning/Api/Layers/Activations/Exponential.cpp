#include "DeepLearning/Api/Layers/Activations/Exponential.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("exponential", &Thor::Exponential::deserialize);
    return true;
}();
}
