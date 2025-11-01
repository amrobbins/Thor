#include "DeepLearning/Api/Layers/Activations/Softmax.h"

using namespace Thor;

namespace {
[[maybe_unused]] static bool registered = []() {
    Activation::registry["softmax"] = &Softmax::deserialize;
    return true;
}();
}
