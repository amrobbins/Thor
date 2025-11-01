#include "DeepLearning/Api/Layers/Activations/Gelu.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["gelu"] = &Gelu::deserialize;
    return true;
}();
}
