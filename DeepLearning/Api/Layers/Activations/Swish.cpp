#include "DeepLearning/Api/Layers/Activations/Swish.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["swish"] = &Swish::deserialize;
    return true;
}();
}
