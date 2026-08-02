#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
           uint64_t maxTotalValues,
           uint64_t batchSize,
           DataType offsetsDataType) {
            if (name.empty()) {
                throw nb::value_error("RaggedNetworkInput name must not be empty.");
            }
            if (maxTotalValues == 0) {
                throw nb::value_error("RaggedNetworkInput max_total_values must be >= 1.");
            }
            for (uint64_t dim : trailingDimensions) {
                if (dim == 0) {
                    throw nb::value_error("RaggedNetworkInput trailing_dimensions must contain only positive dimensions.");
                }
            }
            return RaggedNetworkInput::Builder()
                .network(network)
                .name(name)
                .valuesDataType(valuesDataType)
                .offsetsDataType(offsetsDataType)
                .trailingDimensions(trailingDimensions)
                .maxTotalValues(maxTotalValues)
                .batchSize(batchSize)
                .build();
        },
        "network"_a,
        "name"_a,
        "values_data_type"_a,
        "trailing_dimensions"_a,
        "max_total_values"_a,
        "batch_size"_a,
        "offsets_data_type"_a = DataType::UINT32,
        R"nbdoc(
Create one logical external ragged network input and return its ``thor.RaggedTensor`` handle.

Thor creates the physical ``<name>.values`` and ``<name>.offsets`` NetworkInput
layers internally. Callers bind datasets and layers to the logical ``name``;
the physical pair is an implementation detail.
        )nbdoc");
}
