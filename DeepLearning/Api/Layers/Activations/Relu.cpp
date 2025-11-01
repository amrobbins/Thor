#include "DeepLearning/Api/Layers/Activations/Relu.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["relu"] = &Relu::deserialize;
    return true;
}();
}
