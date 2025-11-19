#pragma once
#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Tensor/Tensor.h"

using TensorList = nanobind::typed<nanobind::list, Thor::Tensor>;
