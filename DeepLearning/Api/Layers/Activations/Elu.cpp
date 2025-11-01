#include "DeepLearning/Api/Layers/Activations/Elu.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["elu"] = &Elu::deserialize;
    return true;
}();
}
