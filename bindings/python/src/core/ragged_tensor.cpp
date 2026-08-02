#include <nanobind/nanobind.h>

#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;
using DataType = ThorImplementation::DataType;

void bind_ragged_tensor(nb::module_ &thor) {
    auto ragged = nb::class_<RaggedTensor>(thor, "RaggedTensor");
    ragged.attr("__module__") = "thor";

    ragged.def(nb::init<Tensor, Tensor>(), "values"_a, "offsets"_a,
               R"nbdoc(
Construct a logical rank-1 ragged tensor from packed values and canonical row-partition offsets.

``values`` has shape ``[max_total_values, *trailing_dimensions]`` and ``offsets``
has shape ``[batch_size + 1]`` with dtype ``uint32`` or ``uint64``. Offsets are
structural metadata; they are not differentiable model values.
               )nbdoc");
    ragged.def(
        nb::init<DataType, const std::vector<uint64_t> &, uint64_t, uint64_t, DataType>(),
        "values_data_type"_a,
        "trailing_dimensions"_a,
        "batch_size"_a,
        "max_total_values"_a,
        "offsets_data_type"_a = ThorImplementation::kDefaultRowPartitionOffsetDataType,
        R"nbdoc(
Construct a logical ragged tensor descriptor using Thor's canonical packed representation.
               )nbdoc");

    ragged.def_prop_ro("values", &RaggedTensor::getValues);
    ragged.def_prop_ro("offsets", &RaggedTensor::getOffsets);
    ragged.def_prop_ro("values_data_type", &RaggedTensor::getValuesDataType);
    ragged.def_prop_ro("offsets_data_type", &RaggedTensor::getOffsetsDataType);
    ragged.def_prop_ro("trailing_dimensions", &RaggedTensor::getTrailingDimensions);
    ragged.def_prop_ro("batch_size", &RaggedTensor::getBatchSize);
    ragged.def_prop_ro("max_total_values", &RaggedTensor::getMaxTotalValues);
    ragged.def_prop_ro("ragged_rank", &RaggedTensor::getRaggedRank);
    ragged.def("get_values", &RaggedTensor::getValues);
    ragged.def("get_offsets", &RaggedTensor::getOffsets);
    ragged.def("get_values_data_type", &RaggedTensor::getValuesDataType);
    ragged.def("get_offsets_data_type", &RaggedTensor::getOffsetsDataType);
    ragged.def("get_trailing_dimensions", &RaggedTensor::getTrailingDimensions);
    ragged.def("get_batch_size", &RaggedTensor::getBatchSize);
    ragged.def("get_max_total_values", &RaggedTensor::getMaxTotalValues);
    ragged.def("get_ragged_rank", &RaggedTensor::getRaggedRank);
    ragged.def("get_id", &RaggedTensor::getId);
    ragged.def("__eq__", [](const RaggedTensor &a, const RaggedTensor &b) { return a == b; }, "other"_a);
    ragged.def("__ne__", [](const RaggedTensor &a, const RaggedTensor &b) { return a != b; }, "other"_a);
    ragged.def("version", &RaggedTensor::getVersion);
}
