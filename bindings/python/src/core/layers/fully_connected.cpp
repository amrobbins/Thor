#include <nanobind/nanobind.h>

#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <nanobind/stl/optional.h>

#include "bindings/python/src/core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Thor::Tensor::DataType;

Activation::Builder *getDefaultFCActivation() {
    static Relu::Builder defaultActivation;
    return &defaultActivation;
}

void bind_fully_connected(nb::module_ &m) {
    nb::class_<FullyConnected, TrainableWeightsBiasesLayer>(m, "FullyConnected")
        .def(
            "__init__",
            [](FullyConnected *self,
               Network &network,
               Tensor featureInput,
               uint32_t numOutputFeatures,
               bool hasBias,
               optional<Activation::Builder *> activation,
               Initializer::Builder *weights_initializer,
               Initializer::Builder *biases_initializer,
               bool add_drop_out,
               float drop_proportion,
               bool add_batch_normalization,
               float batch_norm_exp_running_avg_factor,
               float batch_norm_epsilon) {
                FullyConnected::Builder builder;
                builder.network(network).featureInput(featureInput).numOutputFeatures(numOutputFeatures).hasBias(hasBias);

                if (!activation.has_value())
                    builder.noActivation();  // Explicitly no activation applied
                else
                    builder.activationBuilder(*(activation.value()));

                if (weights_initializer != nullptr)
                    builder.weightsInitializerBuilder(*weights_initializer);
                if (biases_initializer != nullptr)
                    builder.biasInitializerBuilder(*biases_initializer);

                if (add_drop_out)
                    builder.dropOut(drop_proportion);
                if (add_batch_normalization)
                    builder.batchNormalization(batch_norm_exp_running_avg_factor, batch_norm_epsilon);

                FullyConnected built = builder.build();

                new (self) FullyConnected(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "num_output_features"_a,
            "has_bias"_a = true,
            nb::arg("activation").none() = getDefaultFCActivation(),
            "weights_initializer"_a = nb::none(),
            "biases_initializer"_a = nb::none(),
            "add_drop_out"_a = false,
            "drop_proportion"_a = 0.0f,
            "add_batch_normalization"_a = false,
            "batch_norm_exp_running_avg_factor"_a = 0.05f,
            "batch_norm_epsilon"_a = 1e-4f,
            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_input: thor.Tensor, "
                    "num_output_features: int, "
                    "has_bias: bool = True, "
                    "activation: thor.Activation | None = thor.activations.Relu(), "
                    "weights_initializer: thor.initializers.Initializer = thor.initializers.Glorot(), "
                    "biases_initializer: thor.initializers.Initializer = thor.initializers.Glorot(), "
                    "add_drop_out: bool = False, "
                    "drop_proportion: float = 0.0, "
                    "add_batch_normalization: bool = False, "
                    "batch_norm_exp_running_avg_factor: float = 0.05, "
                    "batch_norm_epsilon: float = 1e-4"
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
        add_drop_out : bool, default False
            If True, inserts a DropOut layer, sequenced as described above.
        drop_proportion : float, default 0.0
            Fraction of units to drop when dropout is enabled.
            Ignored if ``add_drop_out`` is False.
        add_batch_normalization : bool, default False
            If True, inserts a batch-normalization layer, sequenced as
            described above.
        batch_norm_exp_running_avg_factor : float, default 0.05
            Exponential running-average factor used to update the batch
            normalization mean and variance statistics during training.
            Ignored if ``add_batch_normalization`` is False.
        batch_norm_epsilon : float, default 1e-4
            Small constant added to the variance in batch normalization
            for numerical stability.
            Ignored if ``add_batch_normalization`` is False.
        )nbdoc");
}
