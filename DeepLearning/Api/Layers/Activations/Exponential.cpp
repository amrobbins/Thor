#include "DeepLearning/Api/Layers/Activations/Exponential.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["exponential"] = &Exponential::deserialize;
    return true;
}();
}
