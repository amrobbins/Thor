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
                if (offsetsDataType.has_value() || maxValuesPerRow.has_value()) {
                    throw nb::value_error(
                        "RaggedNetworkInput partition=... is the sole structural source of truth; "
                        "do not also specify offsets_data_type or max_values_per_row.");
                }
                builder.partition(partition.value());
            } else {
                if (!maxValuesPerRow.has_value()) {
                    throw nb::value_error(
                        "RaggedNetworkInput requires a semantic max_values_per_row for an independent partition; "
                        "placement batch size and packed capacity cannot define model geometry.");
                }
                if (offsetsDataType.has_value()) builder.offsetsDataType(offsetsDataType.value());
                builder.maxValuesPerRow(maxValuesPerRow.value());
            }
            return builder.build();
        },
        "network"_a,
        "name"_a,
        "values_data_type"_a,
        "trailing_dimensions"_a,
        "offsets_data_type"_a = nb::none(),
        "max_values_per_row"_a = nb::none(),
        "partition"_a = nb::none(),
        R"nbdoc(
Create one logical external ragged network input and return its ``thor.RaggedTensor`` handle.

Without ``partition``, provide semantic ``max_values_per_row`` (M).
The model holds logical values ``[M, ...trailing_dimensions]`` and one
canonical partition token ``[1]``. Placement with batch B derives physical
packed values capacity ``B * M``. Runtime batch size and total packed capacity
are not arguments to this logical model constructor.
``offsets_data_type`` defaults to uint32.

With ``partition=<existing_ragged_tensor>``, Thor creates only
``<name>.values`` and reuses the exact canonical row partition of the referenced
logical RaggedNetworkInput. Structural arguments must not be repeated in this
form. At direct inference time, a shared-partition logical input may be supplied
as just a packed ``thor.physical.PhysicalTensor`` under ``name``; its partition
comes from the referenced logical ragged input for that batch.
        )nbdoc");
}
