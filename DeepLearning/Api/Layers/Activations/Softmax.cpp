#include "DeepLearning/Api/Layers/Activations/Softmax.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("softmax", &Thor::Softmax::deserialize);
    return true;
}();
}