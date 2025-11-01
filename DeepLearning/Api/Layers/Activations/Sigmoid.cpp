#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["sigmoid"] = &Sigmoid::deserialize;
    return true;
}();
}
