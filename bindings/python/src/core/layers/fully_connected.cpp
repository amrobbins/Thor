#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Tensor::DataType;

void bind_fully_connected(nb::module_ &m) {
    nb::class_<FullyConnected, TrainableWeightsBiasesLayer>(m, "FullyConnected")
        .def(
            "__init__",
            [](FullyConnected *self,
               Network &network,
               Tensor featureInput,
               uint32_t numOutputFeatures,
               bool hasBias,
               shared_ptr<Activation> activation,
               shared_ptr<Initializer> weights_initializer,
               shared_ptr<Initializer> biases_initializer) {
                FullyConnected::Builder builder;
                builder.network(network).featureInput(featureInput).numOutputFeatures(numOutputFeatures).hasBias(hasBias);

                if (activation == nullptr) {
                    // FIXME: This does not work, need to find the right pattern for this.
                    printf("\nA\n");
                    builder.noActivation();
                } else {
                    printf("\nB\n");
                    builder.activation(activation);
                }

                if (weights_initializer != nullptr)
                    builder.weightsInitializer(weights_initializer);
                if (biases_initializer != nullptr)
                    builder.biasInitializer(biases_initializer);

                FullyConnected built = builder.build();

                new (self) FullyConnected(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "num_output_features"_a,
            "has_bias"_a = true,
            "activation"_a = Relu(),
            "weights_initializer"_a = nb::none(),
            "biases_initializer"_a = nb::none(),
            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_input: thor.Tensor, "
                    "num_output_features: int, "
                    "has_bias: bool = True, "
                    "activation: thor.Activation | None = thor.activations.Relu(), "
                    "weights_initializer: thor.initializers.Initializer = thor.initializers.Glorot(), "
                    "biases_initializer: thor.initializers.Initializer = thor.initializers.Glorot() "
                    ") -> None"),

            R"nbdoc(
        Fully connected (dense) layer.

        Builds a fully connected layer with optional activation, dropout,
        and batch normalization. This is the standard affine layer

            y = W x + b

        optionally preceded by batch normalization and or drop out,
        and optionally followed by a non-linear activation.

        The connection order of the optional layers, when used, is the following:
        [batch norm] -> [drop out] -> [fully connected] -> [activation]

        Parameters
        ----------
        network : thor.Network
            The network that the layer should be added to.
        feature_input : thor.Tensor
            Input feature tensor for this layer.
        num_output_features : int
            Number of output features (units) produced by this layer.
        has_bias : bool, default True
            Whether to learn an additive bias term.
        activation : thor.Activation or None, default thor.activations.Relu()
            Activation to apply after the linear transform (and optional
            batch normalization). Pass ``None`` not use any activation and
            keep the layer purely linear.
        weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the weight matrix.
        biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the bias vector.
        )nbdoc")
        .def(
            "get_feature_output",
            [](FullyConnected &self) -> Tensor {
                Optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
                return maybeFeatureOutput.get();
            },
            nb::sig("def get_feature_output(self) -> Optional[thor.Tensor]"),
            R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");
}
