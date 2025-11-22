#include <nanobind/nanobind.h>

#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
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

Activation::Builder *getDefaultConvActivation() {
    static Relu::Builder defaultActivation;
    return &defaultActivation;
}

void bind_convolution_2d(nb::module_ &m) {
    nb::class_<Convolution2d, TrainableWeightsBiasesLayer>(m, "Convolution2d")
        .def(
            "__init__",
            [](Convolution2d *self,
               Network &network,
               Tensor featureInput,
               uint32_t numOutputChannels,
               uint32_t filterHeight,
               uint32_t filterWidth,
               uint32_t verticalStride,
               uint32_t horizontalStride,
               uint32_t verticalPadding,
               uint32_t horizontalPadding,
               bool samePadding,
               bool verticalSamePadding,
               bool horizontalSamePadding,
               bool hasBias,
               optional<Activation::Builder *> activation,
               Initializer::Builder *weights_initializer,
               Initializer::Builder *biases_initializer,
               bool add_drop_out,
               float drop_proportion,
               bool add_batch_normalization,
               float batch_norm_exp_running_avg_factor,
               float batch_norm_epsilon) {
                Convolution2d::Builder builder;
                builder.network(network)
                    .featureInput(featureInput)
                    .numOutputChannels(numOutputChannels)
                    .filterHeight(filterHeight)
                    .filterWidth(filterWidth)
                    .verticalStride(verticalStride)
                    .horizontalStride(horizontalStride)
                    .hasBias(hasBias);

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

                // These can't all be specified at the same time, but logic to enforce that
                // is on the implementation side, not the binding side.
                if (verticalPadding != 0)
                    builder.verticalPadding(verticalPadding);
                if (horizontalPadding != 0)
                    builder.horizontalPadding(horizontalPadding);
                if (samePadding)
                    builder.samePadding();
                if (verticalSamePadding)
                    builder.verticalSamePadding();
                if (horizontalSamePadding)
                    builder.horizontalSamePadding();

                Convolution2d built = builder.build();

                new (self) Convolution2d(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "num_output_channels"_a,
            "filter_height"_a,
            "filter_width"_a,
            "vertical_stride"_a = 1,
            "horizontal_stride"_a = 1,
            "vertical_padding"_a = 0,
            "horizontal_padding"_a = 0,
            "same_padding"_a = false,
            "vertical_same_padding"_a = false,
            "horizontal_same_padding"_a = false,
            "has_bias"_a = true,
            nb::arg("activation").none() = getDefaultConvActivation(),
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
                    "num_output_channels: int, "
                    "filter_height: int, "
                    "filter_width: int, "
                    "vertical_stride: int = 1, "
                    "horizontal_stride: int = 1, "
                    "vertical_padding: int = 0, "
                    "horizontal_padding: int = 0, "
                    "same_padding: bool = False, "
                    "vertical_same_padding: bool = False, "
                    "horizontal_same_padding: bool = False, "
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
        2D convolution layer.

        Builds a trainable 2D convolutional layer with optional activation,
        dropout, and batch normalization. This layer applies a bank of
        learnable filters over the spatial dimensions of the input tensor.

        The connection order of the optional layers, when used, is the following:
        [batch norm] -> [drop out] -> [fully connected] -> [activation]

        Parameters
        ----------
        network : thor.Network
            The network that the layer should be added to. The network
            owns the layer and manages its lifetime.
        feature_input : thor.Tensor
            Input feature tensor for this layer.
            Expected layout matches the underlying Thor tensor convention
            of CHW on the API side. The physical implementation side adds
            the batch layer and uses NCHW.
        num_output_channels : int
            Number of output channels produced by the layer.
        filter_height : int
            Height of each convolution filter (kernel size in the vertical
            dimension).
        filter_width : int
            Width of each convolution filter (kernel size in the horizontal
            dimension).
        vertical_stride : int, default 1
            Stride of the convolution in the vertical direction.
        horizontal_stride : int, default 1
            Stride of the convolution in the horizontal direction.
        vertical_padding : int, default 0
            Amount of explicit zero-padding to apply above and below the
            input. Cannot be specified when vertical_same_padding is enabled.
        horizontal_padding : int, default 0
            Amount of explicit zero-padding to apply left and right of the
            input. Cannot be specified when horizontal_same_padding is enabled.
        same_padding : bool, default False
            If True, apply symmetric “same” padding in both directions so
            that the output spatial size matches the input (subject to
            stride constraints). When same_padding is enabled, do not specify
            any other padding parameters.
        vertical_same_padding : bool, default False
            If True, apply “same” padding in the vertical dimension.
            When vertical_same_padding is specified, do not specify any other
            vertical padding parameters.
        horizontal_same_padding : bool, default False
            If True, apply “same” padding in the horizontal dimension.
            When horizontal_same_padding is specified, do not specify any other
            horizontal padding parameters.
        has_bias : bool, default True
            Whether to learn an additive bias per output channel.
        activation : thor.Activation or None, default thor.activations.Relu()
            Activation to apply after the convolution
            Pass ``None`` to not use any activation and keep the layer
            purely linear.
        weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the convolution kernel weights.
        biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
            Initializer for the bias vector.
        add_drop_out : bool, default False
            If True, inserts a DropOut layer, sequenced as described above.
        drop_proportion : float, default 0.0
            Fraction of activations to drop when dropout is enabled.
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
