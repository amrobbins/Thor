#include "DeepLearning/Api/Layers/Activations/Relu.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["relu"] = &Relu::deserialize;
    return true;
}();
}
