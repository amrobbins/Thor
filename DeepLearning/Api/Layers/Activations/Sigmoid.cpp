#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["sigmoid"] = &Sigmoid::deserialize;
    return true;
}();
}
