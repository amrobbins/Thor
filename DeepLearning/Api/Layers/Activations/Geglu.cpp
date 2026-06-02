#include "DeepLearning/Api/Layers/Activations/Geglu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("geglu", &Thor::Geglu::deserialize);
    return true;
}();
}  // namespace
