#include "DeepLearning/Api/Layers/Activations/Swiglu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("swiglu", &Thor::Swiglu::deserialize);
    return true;
}();
}  // namespace
