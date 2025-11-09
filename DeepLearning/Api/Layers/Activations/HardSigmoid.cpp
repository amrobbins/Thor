#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("hard_sigmoid", &Thor::HardSigmoid::deserialize);
    return true;
}();
}
