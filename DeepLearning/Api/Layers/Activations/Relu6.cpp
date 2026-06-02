#include "DeepLearning/Api/Layers/Activations/Relu6.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("relu6", &Thor::Relu6::deserialize);
    return true;
}();
}  // namespace
