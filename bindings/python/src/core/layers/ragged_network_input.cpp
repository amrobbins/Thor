#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;
using DataType = ThorImplementation::DataType;

void bind_ragged_network_input(nb::module_ &m) {
    m.def(
        "RaggedNetworkInput",
        [](Network &network,
           const std::string &name,
           DataType valuesDataType,
           const std::vector<uint64_t> &trailingDimensions,
           std::optional<uint64_t> maxTotalValues,
           std::optional<uint64_t> batchSize,
           std::optional<DataType> offsetsDataType,
           std::optional<uint64_t> maxValuesPerRow,
           std::optional<RaggedTensor> partition) {
            if (name.empty()) {
                throw nb::value_error("RaggedNetworkInput name must not be empty.");
            }
            for (uint64_t dim : trailingDimensions) {
                if (dim == 0) {
                    throw nb::value_error("RaggedNetworkInput trailing_dimensions must contain only positive dimensions.");
                }
            }

            RaggedNetworkInput::Builder builder = RaggedNetworkInput::Builder()
                .network(network)
                .name(name)
                .valuesDataType(valuesDataType)
                .trailingDimensions(trailingDimensions);

            if (partition.has_value()) {
                if (maxTotalValues.has_value() || batchSize.has_value() || offsetsDataType.has_value() ||
                    maxValuesPerRow.has_value()) {
                    throw nb::value_error(
                        "RaggedNetworkInput partition=... is the sole structural source of truth; "
                        "do not also specify max_total_values, batch_size, offsets_data_type, or max_values_per_row.");
                }
                builder.partition(partition.value());
            } else {
                if (!maxTotalValues.has_value() || maxTotalValues.value() == 0) {
                    throw nb::value_error("RaggedNetworkInput max_total_values must be >= 1 when partition is not provided.");
                }
                if (!batchSize.has_value()) {
                    throw nb::value_error("RaggedNetworkInput batch_size is required when partition is not provided.");
                }
                builder.maxTotalValues(maxTotalValues.value()).batchSize(batchSize.value());
                if (offsetsDataType.has_value()) builder.offsetsDataType(offsetsDataType.value());
                if (maxValuesPerRow.has_value()) builder.maxValuesPerRow(maxValuesPerRow.value());
            }
            return builder.build();
        },
        "network"_a,
        "name"_a,
        "values_data_type"_a,
        "trailing_dimensions"_a,
        "max_total_values"_a = nb::none(),
        "batch_size"_a = nb::none(),
        "offsets_data_type"_a = nb::none(),
        "max_values_per_row"_a = nb::none(),
        "partition"_a = nb::none(),
        R"nbdoc(
Create one logical external ragged network input and return its ``thor.RaggedTensor`` handle.

Without ``partition``, Thor creates the physical ``<name>.values`` and
``<name>.offsets`` NetworkInput layers internally. ``max_total_values`` and
``batch_size`` are required and ``offsets_data_type`` defaults to uint32.

With ``partition=<existing_ragged_tensor>``, Thor creates only
``<name>.values`` and reuses the exact canonical row partition of the referenced
logical RaggedNetworkInput. Structural arguments must not be repeated in this
form. At direct inference time, a shared-partition logical input may be supplied
as just a packed ``thor.physical.PhysicalTensor`` under ``name``; its partition
comes from the referenced logical ragged input for that batch.
        )nbdoc");
}
