#include "DeepLearning/Api/Layers/Activations/Elu.h"

#include <stdio.h>

namespace {
static bool Elu_registered = []() {
    Thor::Activation::registry["elu"] = &Thor::Elu::deserialize;
    return true;
}();
}
