#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <optional>
#include <utility>

#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_physical_ragged_tensor(nb::module_& physical) {
    auto ragged = nb::class_<ThorImplementation::RaggedTensor>(physical, "PhysicalRaggedTensor");
    ragged.attr("__module__") = "thor.physical";

    ragged.def(
        "__init__",
        [](ThorImplementation::RaggedTensor* self,
           ThorImplementation::Tensor values,
           ThorImplementation::Tensor offsets,
           std::optional<uint64_t> maxValuesPerRow) {
            if (maxValuesPerRow.has_value()) {
                if (maxValuesPerRow.value() == 0) {
                    throw nb::value_error("PhysicalRaggedTensor max_values_per_row must be >= 1 when provided.");
                }
                new (self) ThorImplementation::RaggedTensor(
                    std::move(values), std::move(offsets), maxValuesPerRow.value());
            } else {
                new (self) ThorImplementation::RaggedTensor(std::move(values), std::move(offsets));
            }
        },
        "values"_a,
        "offsets"_a,
        "max_values_per_row"_a = nb::none(),
        R"nbdoc(
Construct a physical ragged tensor from packed values and canonical row offsets.

``max_values_per_row`` is optional placement-time structural metadata. Supply it
when the logical network input declares the same bound (for example ragged
causal Conv1D). CPU offsets remain the semantic source of runtime row extents;
Thor does not introduce an implicit device-to-host synchronization for GPU
resident offsets.
        )nbdoc");
    ragged.def_prop_ro("values", &ThorImplementation::RaggedTensor::getValues);
    ragged.def_prop_ro("offsets", &ThorImplementation::RaggedTensor::getOffsets);
    ragged.def_prop_ro("batch_size", &ThorImplementation::RaggedTensor::getBatchSize);
    ragged.def_prop_ro("max_total_values", &ThorImplementation::RaggedTensor::getMaxTotalValues);
    ragged.def_prop_ro("max_values_per_row", [](const ThorImplementation::RaggedTensor& value) -> std::optional<uint64_t> {
        if (!value.hasMaxValuesPerRow()) return std::nullopt;
        return value.getMaxValuesPerRow();
    });
    ragged.def_prop_ro("values_data_type", &ThorImplementation::RaggedTensor::getValuesDataType);
    ragged.def_prop_ro("offsets_data_type", &ThorImplementation::RaggedTensor::getOffsetsDataType);
    ragged.def("get_values", &ThorImplementation::RaggedTensor::getValues);
    ragged.def("get_offsets", &ThorImplementation::RaggedTensor::getOffsets);
}
