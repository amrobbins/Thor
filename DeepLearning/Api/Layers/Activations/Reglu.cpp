#include "DeepLearning/Api/Layers/Activations/Reglu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("reglu", &Thor::Reglu::deserialize);
    return true;
}();
}  // namespace
