#include "DeepLearning/Api/Layers/Activations/Exponential.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["exponential"] = &Exponential::deserialize;
    return true;
}();
}
