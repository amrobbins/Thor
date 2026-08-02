#pragma once

#include <nanobind/nanobind.h>

namespace Thor::PythonBindings {

struct PythonRaggedBatch {
    nanobind::object values;
    nanobind::object offsets;
};

}  // namespace Thor::PythonBindings
