#include "DeepLearning/Api/Layers/Activations/Swish.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("swish", &Thor::Swish::deserialize);
    return true;
}();
}
