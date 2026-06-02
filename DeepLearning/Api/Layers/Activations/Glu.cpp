#include "DeepLearning/Api/Layers/Activations/Glu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("glu", &Thor::Glu::deserialize);
    return true;
}();
}  // namespace
