#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("sigmoid", &Thor::Sigmoid::deserialize);
    return true;
}();
}  // namespace
