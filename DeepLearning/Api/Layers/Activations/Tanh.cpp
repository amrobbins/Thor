#include "DeepLearning/Api/Layers/Activations/Tanh.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["tanh"] = &Tanh::deserialize;
    return true;
}();
}
