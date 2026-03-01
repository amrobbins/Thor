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

using DataType = ThorImplementation::TensorDescriptor::DataType;

void bind_fully_connected(nb::module_ &m) {
    auto fully_connected = nb::class_<FullyConnected, TrainableWeightsBiasesLayer>(m, "FullyConnected");
    fully_connected.attr("__module__") = "thor.layers";

    fully_connected.def(
        "__init__",
        [](FullyConnected *self,
           Network &network,
           Tensor featureInput,
           uint32_t numOutputFeatures,
           bool hasBias,
           shared_ptr<Activation> activation,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer) {
            if (numOutputFeatures == 0) {
                throw nb::value_error("FullyConnected instance: num_output_features must be > 0.");
            }

            FullyConnected::Builder builder;
            builder.network(network).featureInput(featureInput).numOutputFeatures(numOutputFeatures).hasBias(hasBias);

            if (activation == nullptr) {
                builder.noActivation();
            } else {
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
        "activation"_a.none() = nb::none(),
        "weights_initializer"_a.none() = nb::none(),
        "biases_initializer"_a.none() = nb::none());

    fully_connected.def(
        "get_feature_output",
        [](FullyConnected &self) -> Tensor {
            Optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.get();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    fully_connected.attr("__doc__") = R"nbdoc(
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
        )nbdoc";
}
