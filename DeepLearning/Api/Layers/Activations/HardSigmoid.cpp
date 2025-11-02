#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["hard_sigmoid"] = &HardSigmoid::deserialize;
    return true;
}();
}
