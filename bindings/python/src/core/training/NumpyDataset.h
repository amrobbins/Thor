#pragma once

#include <nanobind/nanobind.h>

namespace Thor::PythonBindings {

void bindNumpyDataset(nanobind::module_ &training);

}  // namespace Thor::PythonBindings
