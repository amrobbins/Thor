#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["soft_plus"] = &SoftPlus::deserialize;
    return true;
}();
}
