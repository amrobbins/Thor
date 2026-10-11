#include <nanobind/nanobind.h>

#include <nanobind/stl/vector.h>

#include <new>
#include <utility>

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

    ragged.def(
        "__init__",
        [](RaggedTensor* self, Tensor values, Tensor rowPartitionToken, uint64_t maxValuesPerRow) {
            if (maxValuesPerRow == 0) {
                throw nb::value_error("RaggedTensor max_values_per_row (semantic M) must be >= 1.");
            }
            if (values.getDimensions().empty() || values.getDimensions().front() != maxValuesPerRow) {
                throw nb::value_error("RaggedTensor values must have leading logical dimension max_values_per_row (M).");
            }
            if (rowPartitionToken.getDimensions() != std::vector<uint64_t>{1}) {
                throw nb::value_error("RaggedTensor row_partition_token must have logical shape [1], not physical [B+1] offsets.");
            }
            if (!ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(rowPartitionToken.getDataType())) {
                throw nb::value_error("RaggedTensor row_partition_token must use uint32 or uint64 dtype.");
            }
            new (self) RaggedTensor(RaggedTensor::logical(
                std::move(values), std::move(rowPartitionToken), maxValuesPerRow));
        },
        "values"_a,
        "row_partition_token"_a,
        "max_values_per_row"_a,
        R"nbdoc(
Construct a logical rank-1 ragged tensor with explicit semantic M and a canonical
row-partition token. The values have shape ``[M, *trailing_dimensions]`` and the
token has shape ``[1]`` with dtype uint32 or uint64. Neither is a packed runtime
buffer: placement determines B, physical ``[B*M, ...]`` capacity and, if required,
``[B+1]`` offsets. To reuse a partition across values, pass the same token.
        )nbdoc");

    ragged.def_prop_ro("values", &RaggedTensor::getValues);
    ragged.def_prop_ro("offsets", &RaggedTensor::getOffsets);  // structural-token compatibility alias
    ragged.def_prop_ro("row_partition_token", &RaggedTensor::getRowPartitionToken);
    ragged.def_prop_ro("row_partition_id", &RaggedTensor::getRowPartitionId);
    ragged.def_prop_ro("values_data_type", &RaggedTensor::getValuesDataType);
    ragged.def_prop_ro("offsets_data_type", &RaggedTensor::getOffsetsDataType);
    ragged.def_prop_ro("trailing_dimensions", &RaggedTensor::getTrailingDimensions);
    ragged.def_prop_ro("max_values_per_row", &RaggedTensor::getMaxValuesPerRow);
    ragged.def_prop_ro("ragged_rank", &RaggedTensor::getRaggedRank);
    ragged.def("get_values", &RaggedTensor::getValues);
    ragged.def("get_offsets", &RaggedTensor::getOffsets);
    ragged.def("get_row_partition_token", &RaggedTensor::getRowPartitionToken);
    ragged.def("get_row_partition_id", &RaggedTensor::getRowPartitionId);
    ragged.def("shares_partition_with", &RaggedTensor::sharesPartitionWith, "other"_a);
    ragged.def("get_values_data_type", &RaggedTensor::getValuesDataType);
    ragged.def("get_offsets_data_type", &RaggedTensor::getOffsetsDataType);
    ragged.def("get_trailing_dimensions", &RaggedTensor::getTrailingDimensions);
    ragged.def("get_ragged_rank", &RaggedTensor::getRaggedRank);
    ragged.def("get_id", &RaggedTensor::getId);
    ragged.def("__eq__", [](const RaggedTensor &a, const RaggedTensor &b) { return a == b; }, "other"_a);
    ragged.def("__ne__", [](const RaggedTensor &a, const RaggedTensor &b) { return a != b; }, "other"_a);
    ragged.def("version", &RaggedTensor::getVersion);
}
