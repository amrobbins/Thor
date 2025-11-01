#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["hardSigmoid"] = &HardSigmoid::deserialize;
    return true;
}();
}
