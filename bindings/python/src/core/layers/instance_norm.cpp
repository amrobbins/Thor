#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/InstanceNorm.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

using DataType = ThorImplementation::DataType;

void bind_instance_norm(nb::module_& m) {
    auto instance_norm = nb::class_<InstanceNorm, TrainableLayer>(m, "InstanceNorm");
    instance_norm.attr("__module__") = "thor.layers";

    instance_norm.def(
        "__init__",
        [](InstanceNorm* self,
           Network& network,
           Tensor feature_input,
           double epsilon,
           DataType parameter_data_type,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           shared_ptr<Optimizer> weights_optimizer,
           shared_ptr<Optimizer> biases_optimizer) {
            if (!(epsilon > 0.0)) {
                throw nb::value_error("InstanceNorm instance: epsilon must be > 0.");
            }

            InstanceNorm::Builder builder;
            builder.network(network).featureInput(feature_input).epsilon(epsilon).parameterDataType(parameter_data_type);
            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            if (biases_initializer != nullptr)
                builder.biasesInitializer(biases_initializer);
            builder.weightsOptimizer(weights_optimizer);
            builder.biasesOptimizer(biases_optimizer);

            new (self) InstanceNorm(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "epsilon"_a = 1.0e-5,
        "parameter_data_type"_a = DataType::FP32,
        "weights_initializer"_a.none() = nb::none(),
        "biases_initializer"_a.none() = nb::none(),
        "weights_optimizer"_a.none() = nb::none(),
        "biases_optimizer"_a.none() = nb::none());

    instance_norm.def(
        "get_feature_output",
        [](InstanceNorm& self) -> Tensor {
            optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(Return the output tensor produced by this layer.)nbdoc");

    instance_norm.def("get_channel_count", [](InstanceNorm& self) { return self.getChannelCount(); });
    instance_norm.def("get_epsilon", [](InstanceNorm& self) { return self.getEpsilon(); });
    instance_norm.def("get_parameter_data_type", [](InstanceNorm& self) { return self.getParameterDataType(); });

    instance_norm.attr("__doc__") = R"nbdoc(
        Instance normalization over each sample/channel's contiguous spatial region.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_input : thor.Tensor
            Input feature tensor with API dimensions [C, spatial...]. The runtime batch dimension is added by stamping.
        epsilon : float, default 1e-5
            Positive numerical-stability epsilon.
        parameter_data_type : thor.DataType, default thor.DataType.fp32
            Data type for scale and bias. Thor currently requires fp32 for cuDNN Frontend InstanceNorm.
        )nbdoc";
}
