#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace {

vector<uint64_t> normalizedShapeFromPython(const nb::object& obj, const Tensor& featureInput) {
    if (obj.is_none()) {
        const vector<uint64_t> dims = featureInput.getDimensions();
        if (dims.empty()) {
            throw nb::value_error("RMSNorm instance: feature_input must have at least one feature dimension.");
        }
        return {dims.back()};
    }
    return nb::cast<vector<uint64_t>>(obj);
}

}  // namespace

void bind_rms_norm(nb::module_& m) {
    auto rms_norm = nb::class_<RMSNorm, TrainableLayer>(m, "RMSNorm");
    rms_norm.attr("__module__") = "thor.layers";

    rms_norm.def(
        "__init__",
        [](RMSNorm* self,
           Network& network,
           Tensor feature_input,
           nb::object normalized_shape,
           double epsilon,
           DataType parameter_data_type,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Optimizer> weights_optimizer) {
            if (!(epsilon > 0.0)) {
                throw nb::value_error("RMSNorm instance: epsilon must be > 0.");
            }

            vector<uint64_t> shape = normalizedShapeFromPython(normalized_shape, feature_input);
            RMSNorm::Builder builder;
            builder.network(network).featureInput(feature_input).normalizedShape(shape).epsilon(epsilon).parameterDataType(parameter_data_type);
            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            builder.weightsOptimizer(weights_optimizer);

            new (self) RMSNorm(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "normalized_shape"_a.none() = nb::none(),
        "epsilon"_a = 1.0e-5,
        "parameter_data_type"_a = DataType::FP32,
        "weights_initializer"_a.none() = nb::none(),
        "weights_optimizer"_a.none() = nb::none());

    rms_norm.def(
        "get_feature_output",
        [](RMSNorm& self) -> Tensor {
            optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(Return the output tensor produced by this layer.)nbdoc");

    rms_norm.def("get_normalized_shape", [](RMSNorm& self) { return self.getNormalizedShape(); });
    rms_norm.def("get_epsilon", [](RMSNorm& self) { return self.getEpsilon(); });
    rms_norm.def("get_parameter_data_type", [](RMSNorm& self) { return self.getParameterDataType(); });

    rms_norm.attr("__doc__") = R"nbdoc(
        Root Mean Square Layer Normalization over a contiguous trailing normalized shape.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_input : thor.Tensor
            Input feature tensor for this layer.
        normalized_shape : Sequence[int] or None, default None
            Trailing feature dimensions to normalize over. None normalizes the final feature dimension.
        epsilon : float, default 1e-5
            Positive numerical-stability epsilon.
        parameter_data_type : thor.DataType, default thor.DataType.fp32
            Data type for scale weights. Thor currently requires fp32 for cuDNN Frontend RMSNorm.
        )nbdoc";
}
