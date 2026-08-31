#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "bindings/python/src/core/cast.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

using DataType = ThorImplementation::DataType;

namespace {

vector<uint64_t> normalizedShapeFromPython(const nb::object& obj, const Tensor& featureInput) {
    if (obj.is_none()) {
        const vector<uint64_t> dims = featureInput.getDimensions();
        if (dims.empty()) {
            throw nb::value_error("LayerNorm instance: feature_input must have at least one feature dimension.");
        }
        return {dims.back()};
    }
    return pybind::castArgument<vector<uint64_t>>(obj, "LayerNorm", "normalized_shape", "Sequence[int] or None", false);
}

}  // namespace

void bind_layer_norm(nb::module_& m) {
    auto layer_norm = nb::class_<LayerNorm, TrainableLayer>(m, "LayerNorm");
    layer_norm.attr("__module__") = "thor.layers";

    layer_norm.def(
        "__init__",
        [](LayerNorm* self,
           Network& network,
           nb::object feature_input,
           nb::object normalized_shape,
           double epsilon,
           DataType parameter_data_type,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Initializer> biases_initializer,
           shared_ptr<Optimizer> weights_optimizer,
           shared_ptr<Optimizer> biases_optimizer) {
            if (!(epsilon > 0.0)) {
                throw nb::value_error("LayerNorm instance: epsilon must be > 0.");
            }

            Tensor valuesInput;
            LayerNorm::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                RaggedTensor raggedInput = nb::cast<RaggedTensor>(feature_input);
                valuesInput = raggedInput.getValues();
                builder.featureInput(raggedInput);
            } else if (nb::isinstance<Tensor>(feature_input)) {
                valuesInput = nb::cast<Tensor>(feature_input);
                builder.featureInput(valuesInput);
            } else {
                throw nb::type_error("LayerNorm feature_input must be thor.Tensor or thor.RaggedTensor.");
            }

            vector<uint64_t> shape = normalizedShapeFromPython(normalized_shape, valuesInput);
            builder.normalizedShape(shape).epsilon(epsilon).parameterDataType(parameter_data_type);
            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            if (biases_initializer != nullptr)
                builder.biasesInitializer(biases_initializer);
            builder.weightsOptimizer(weights_optimizer);
            builder.biasesOptimizer(biases_optimizer);

            new (self) LayerNorm(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "normalized_shape"_a.none() = nb::none(),
        "epsilon"_a = 1.0e-5,
        "parameter_data_type"_a = DataType::FP32,
        "weights_initializer"_a.none() = nb::none(),
        "biases_initializer"_a.none() = nb::none(),
        "weights_optimizer"_a.none() = nb::none(),
        "biases_optimizer"_a.none() = nb::none());

    layer_norm.def(
        "get_feature_output",
        [](LayerNorm& self) -> nb::object {
            if (optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value()) {
                return nb::cast(raggedOutput.value());
            }
            return nb::cast(self.getFeatureOutput().value());
        },
        R"nbdoc(Return the logical output produced by this layer. Ragged inputs preserve their row partition.)nbdoc");

    layer_norm.def("get_use_ragged", &LayerNorm::getUseRagged);
    layer_norm.def("get_normalized_shape", [](LayerNorm& self) { return self.getNormalizedShape(); });
    layer_norm.def("get_epsilon", [](LayerNorm& self) { return self.getEpsilon(); });
    layer_norm.def("get_parameter_data_type", [](LayerNorm& self) { return self.getParameterDataType(); });

    layer_norm.attr("__doc__") = R"nbdoc(
        Layer normalization over a contiguous trailing normalized shape.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_input : thor.Tensor or thor.RaggedTensor
            Input feature tensor for this layer. Ragged inputs are normalized token-wise over their single trailing channel dimension.
        normalized_shape : Sequence[int] or None, default None
            Trailing feature dimensions to normalize over.  None normalizes the final feature dimension.
        epsilon : float, default 1e-5
            Positive numerical-stability epsilon.
        parameter_data_type : thor.DataType, default thor.DataType.fp32
            Data type for scale and bias.  Thor currently requires fp32 for cuDNN Frontend LayerNorm.
        )nbdoc";
}
