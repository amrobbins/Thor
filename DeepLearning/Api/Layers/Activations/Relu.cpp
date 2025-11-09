#include "DeepLearning/Api/Layers/Activations/Relu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("relu", &Thor::Relu::deserialize);
    return true;
}();
}
