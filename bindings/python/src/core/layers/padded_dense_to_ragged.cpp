#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/PaddedDenseToRagged.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_padded_dense_to_ragged(nb::module_& m) {
    auto layer = nb::class_<PaddedDenseToRagged, MultiConnectionLayer>(m, "PaddedDenseToRagged");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](PaddedDenseToRagged* self, Network& network, Tensor featureInput, RaggedTensor partitionInput) {
            PaddedDenseToRagged built = PaddedDenseToRagged::Builder()
                                            .network(network)
                                            .featureInput(featureInput)
                                            .partitionInput(partitionInput)
                                            .build();
            new (self) PaddedDenseToRagged(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "partition_input"_a,
        R"nbdoc(
Pack a normal padded dense tensor into canonical ragged storage.

``partition_input`` is the sole source of row membership; only its offsets are
consumed and the output reuses that exact partition. The dense input logical
shape is ``[padded_width, *trailing]`` with ``padded_width`` at least
``partition_input.max_values_per_row``. Padding cells are ignored. Backward
materializes dense gradients with exact zeros in padded positions.
)nbdoc");
    layer.def("get_feature_output", &PaddedDenseToRagged::getRaggedFeatureOutput);
    layer.def("get_feature_input", &PaddedDenseToRagged::getDenseFeatureInput);
    layer.def("get_partition_input", &PaddedDenseToRagged::getPartitionInput);
}
