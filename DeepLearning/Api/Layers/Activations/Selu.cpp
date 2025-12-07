#include "DeepLearning/Api/Layers/Activations/Selu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("selu", &Thor::Selu::deserialize);
    return true;
}();
}  // namespace
