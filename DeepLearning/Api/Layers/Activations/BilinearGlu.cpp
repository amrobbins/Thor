#include "DeepLearning/Api/Layers/Activations/BilinearGlu.h"

namespace Thor {}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Activation::register_layer("bilinear_glu", &Thor::BilinearGlu::deserialize);
    return true;
}();
}  // namespace
