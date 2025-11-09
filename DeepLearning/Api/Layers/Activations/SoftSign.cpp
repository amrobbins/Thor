#include "DeepLearning/Api/Layers/Activations/SoftSign.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("soft_sign", &Thor::SoftSign::deserialize);
    return true;
}();
}
