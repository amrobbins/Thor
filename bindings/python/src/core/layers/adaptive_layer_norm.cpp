#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "bindings/python/src/core/cast.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

using DataType = ThorImplementation::DataType;

namespace {

vector<uint64_t> normalizedShapeFromPython(const nb::object& obj, const vector<uint64_t>& featureDimensions) {
    if (obj.is_none()) {
        if (featureDimensions.empty()) {
            throw nb::value_error("AdaptiveLayerNorm instance: feature_input must have at least one feature dimension.");
        }
        return {featureDimensions.back()};
    }
    return pybind::castArgument<vector<uint64_t>>(obj, "AdaptiveLayerNorm", "normalized_shape", "Sequence[int] or None", false);
}

}  // namespace

void bind_adaptive_layer_norm(nb::module_& m) {
    auto adaptive_layer_norm = nb::class_<AdaptiveLayerNorm, MultiConnectionLayer>(m, "AdaptiveLayerNorm");
    adaptive_layer_norm.attr("__module__") = "thor.layers";

    adaptive_layer_norm.def(
        "__init__",
        [](AdaptiveLayerNorm* self,
           Network& network,
           nb::object feature_input,
           Tensor scale_input,
           Tensor bias_input,
           nb::object normalized_shape,
           double epsilon,
           DataType scale_bias_data_type) {
            if (!(epsilon > 0.0)) {
                throw nb::value_error("AdaptiveLayerNorm instance: epsilon must be > 0.");
            }

            AdaptiveLayerNorm::Builder builder;
            builder.network(network);
            vector<uint64_t> featureDimensions;
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                RaggedTensor ragged = nb::cast<RaggedTensor>(feature_input);
                featureDimensions = ragged.getTrailingDimensions();
                builder.featureInput(ragged);
            } else if (nb::isinstance<Tensor>(feature_input)) {
                Tensor dense = nb::cast<Tensor>(feature_input);
                featureDimensions = dense.getDimensions();
                builder.featureInput(dense);
            } else {
                throw nb::type_error("AdaptiveLayerNorm feature_input must be thor.Tensor or thor.RaggedTensor.");
            }

            vector<uint64_t> shape = normalizedShapeFromPython(normalized_shape, featureDimensions);
            builder.scaleInput(scale_input)
                .biasInput(bias_input)
                .normalizedShape(shape)
                .epsilon(epsilon)
                .scaleBiasDataType(scale_bias_data_type);

            new (self) AdaptiveLayerNorm(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "scale_input"_a,
        "bias_input"_a,
        "normalized_shape"_a.none() = nb::none(),
        "epsilon"_a = 1.0e-5,
        "scale_bias_data_type"_a = DataType::FP32);

    adaptive_layer_norm.def(
        "get_feature_output",
        [](AdaptiveLayerNorm& self) -> nb::object {
            if (optional<RaggedTensor> ragged = self.getRaggedFeatureOutput(); ragged.has_value()) {
                return nb::cast(ragged.value());
            }
            optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return nb::cast(maybeFeatureOutput.value());
        },
        R"nbdoc(Return the output tensor produced by this layer. Ragged inputs return a thor.RaggedTensor.)nbdoc");

    adaptive_layer_norm.def("get_data_input", [](AdaptiveLayerNorm& self) -> nb::object {
        if (optional<RaggedTensor> ragged = self.getRaggedDataInput(); ragged.has_value()) {
            return nb::cast(ragged.value());
        }
        return nb::cast(self.getDataInput());
    });
    adaptive_layer_norm.def("get_scale_input", [](AdaptiveLayerNorm& self) { return self.getScaleInput(); });
    adaptive_layer_norm.def("get_bias_input", [](AdaptiveLayerNorm& self) { return self.getBiasInput(); });
    adaptive_layer_norm.def("get_normalized_shape", [](AdaptiveLayerNorm& self) { return self.getNormalizedShape(); });
    adaptive_layer_norm.def("get_epsilon", [](AdaptiveLayerNorm& self) { return self.getEpsilon(); });
    adaptive_layer_norm.def("get_scale_bias_data_type", [](AdaptiveLayerNorm& self) { return self.getScaleBiasDataType(); });
    adaptive_layer_norm.def("get_use_ragged", &AdaptiveLayerNorm::getUseRagged);

    adaptive_layer_norm.attr("__doc__") = R"nbdoc(
        Adaptive layer normalization over a contiguous trailing normalized shape.

        AdaptiveLayerNorm differs from LayerNorm by taking scale and bias as input tensors rather
        than trainable parameters. For a dense feature_input, scale and bias are per-sample affine
        values as before. For a rank-1 RaggedTensor feature_input, scale and bias are dense per-logical-
        row values and Thor broadcasts each row's affine parameters only to that row's active tokens.
        The exact ragged row partition is preserved.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_input : thor.Tensor or thor.RaggedTensor
            Input feature tensor to normalize. Ragged input currently requires exactly one trailing
            channel dimension.
        scale_input : thor.Tensor
            Per-sample/per-logical-row scale tensor. API dimensions must match normalized_shape and
            dtype must be fp32.
        bias_input : thor.Tensor
            Per-sample/per-logical-row bias tensor. API dimensions must match normalized_shape and
            dtype must be fp32.
        normalized_shape : Sequence[int] or None, default None
            Trailing feature dimensions to normalize over. None normalizes the final feature dimension.
        epsilon : float, default 1e-5
            Positive numerical-stability epsilon.
        scale_bias_data_type : thor.DataType, default thor.DataType.fp32
            Data type for scale and bias tensors. Thor currently requires fp32 for cuDNN Frontend AdaLayerNorm.
        )nbdoc";
}
