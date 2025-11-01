#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["softPlus"] = &SoftPlus::deserialize;
    return true;
}();
}
