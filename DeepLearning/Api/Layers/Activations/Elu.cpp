#include "DeepLearning/Api/Layers/Activations/Elu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("elu", &Thor::Elu::deserialize);
    return true;
}();
}
