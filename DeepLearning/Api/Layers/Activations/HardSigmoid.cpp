#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["hardSigmoid"] = &HardSigmoid::deserialize;
    return true;
}();
}
