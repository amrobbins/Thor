#include "DeepLearning/Api/Layers/Activations/Threshold.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("threshold", &Thor::Threshold::deserialize);
    return true;
}();
}  // namespace
