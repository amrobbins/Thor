#include <nanobind/nanobind.h>

#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_physical_ragged_tensor(nb::module_& physical) {
    auto ragged = nb::class_<ThorImplementation::RaggedTensor>(physical, "PhysicalRaggedTensor");
    ragged.attr("__module__") = "thor.physical";

    ragged.def(nb::init<ThorImplementation::Tensor, ThorImplementation::Tensor>(), "values"_a, "offsets"_a);
    ragged.def_prop_ro("values", &ThorImplementation::RaggedTensor::getValues);
    ragged.def_prop_ro("offsets", &ThorImplementation::RaggedTensor::getOffsets);
    ragged.def_prop_ro("batch_size", &ThorImplementation::RaggedTensor::getBatchSize);
    ragged.def_prop_ro("max_total_values", &ThorImplementation::RaggedTensor::getMaxTotalValues);
    ragged.def_prop_ro("max_values_per_row", [](const ThorImplementation::RaggedTensor& value) -> nb::object {
        if (!value.hasMaxValuesPerRow()) return nb::none();
        return nb::int_(value.getMaxValuesPerRow());
    });
    ragged.def_prop_ro("values_data_type", &ThorImplementation::RaggedTensor::getValuesDataType);
    ragged.def_prop_ro("offsets_data_type", &ThorImplementation::RaggedTensor::getOffsetsDataType);
    ragged.def("get_values", &ThorImplementation::RaggedTensor::getValues);
    ragged.def("get_offsets", &ThorImplementation::RaggedTensor::getOffsets);
}
