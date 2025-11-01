#include "DeepLearning/Api/Layers/Activations/SoftSign.h"

using namespace Thor;

namespace {
static bool registered = []() {
    Activation::registry["softSign"] = &SoftSign::deserialize;
    return true;
}();
}
