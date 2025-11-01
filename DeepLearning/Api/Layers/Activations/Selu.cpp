#include "DeepLearning/Api/Layers/Activations/Selu.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["selu"] = &Selu::deserialize;
    return true;
}();
}
