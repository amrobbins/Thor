#include "DeepLearning/Api/Layers/Activations/Selu.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["selu"] = &Selu::deserialize;
    return true;
}();
}
