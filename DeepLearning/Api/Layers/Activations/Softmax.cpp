#include "DeepLearning/Api/Layers/Activations/Softmax.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["softmax"] = &Softmax::deserialize;
    return true;
}();
}
