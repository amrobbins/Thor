#include "DeepLearning/Api/Layers/Activations/SoftSign.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["soft_sign"] = &SoftSign::deserialize;
    return true;
}();
}
