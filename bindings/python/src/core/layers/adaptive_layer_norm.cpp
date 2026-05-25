#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

using DataType = ThorImplementation::DataType;

namespace {

vector<uint64_t> normalizedShapeFromPython(const nb::object& obj, const Tensor& featureInput) {
    if (obj.is_none()) {
        const vector<uint64_t> dims = featureInput.getDimensions();
        if (dims.empty()) {
            throw nb::value_error("AdaptiveLayerNorm instance: feature_input must have at least one feature dimension.");
        }
        return {dims.back()};
    }
    return nb::cast<vector<uint64_t>>(obj);
}

}  // namespace

void bind_adaptive_layer_norm(nb::module_& m) {
    auto adaptive_layer_norm = nb::class_<AdaptiveLayerNorm, MultiConnectionLayer>(m, "AdaptiveLayerNorm");
    adaptive_layer_norm.attr("__module__") = "thor.layers";

    adaptive_layer_norm.def(
        "__init__",
        [](AdaptiveLayerNorm* self,
           Network& network,
           Tensor feature_input,
           Tensor scale_input,
           Tensor bias_input,
           nb::object normalized_shape,
           double epsilon,
           DataType scale_bias_data_type) {
            if (!(epsilon > 0.0)) {
                throw nb::value_error("AdaptiveLayerNorm instance: epsilon must be > 0.");
            }

            vector<uint64_t> shape = normalizedShapeFromPython(normalized_shape, feature_input);
            AdaptiveLayerNorm::Builder builder;
            builder.network(network)
                .featureInput(feature_input)
                .scaleInput(scale_input)
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
        [](AdaptiveLayerNorm& self) -> Tensor {
            optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(Return the output tensor produced by this layer.)nbdoc");

    adaptive_layer_norm.def("get_data_input", [](AdaptiveLayerNorm& self) { return self.getDataInput(); });
    adaptive_layer_norm.def("get_scale_input", [](AdaptiveLayerNorm& self) { return self.getScaleInput(); });
    adaptive_layer_norm.def("get_bias_input", [](AdaptiveLayerNorm& self) { return self.getBiasInput(); });
    adaptive_layer_norm.def("get_normalized_shape", [](AdaptiveLayerNorm& self) { return self.getNormalizedShape(); });
    adaptive_layer_norm.def("get_epsilon", [](AdaptiveLayerNorm& self) { return self.getEpsilon(); });
    adaptive_layer_norm.def("get_scale_bias_data_type", [](AdaptiveLayerNorm& self) { return self.getScaleBiasDataType(); });

    adaptive_layer_norm.attr("__doc__") = R"nbdoc(
        Adaptive layer normalization over a contiguous trailing normalized shape.

        AdaptiveLayerNorm differs from LayerNorm by taking scale and bias as input tensors rather
        than trainable parameters. The scale and bias tensors must match feature_input dimensions and
        are interpreted as per-sample scale/bias values by cuDNN Frontend AdaLayerNorm.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_input : thor.Tensor
            Input feature tensor to normalize.
        scale_input : thor.Tensor
            Per-sample scale tensor. Must have the same dimensions as feature_input and fp32 dtype.
        bias_input : thor.Tensor
            Per-sample bias tensor. Must have the same dimensions as feature_input and fp32 dtype.
        normalized_shape : Sequence[int] or None, default None
            Trailing feature dimensions to normalize over. None normalizes the final feature dimension.
        epsilon : float, default 1e-5
            Positive numerical-stability epsilon.
        scale_bias_data_type : thor.DataType, default thor.DataType.fp32
            Data type for scale and bias tensors. Thor currently requires fp32 for cuDNN Frontend AdaLayerNorm.
        )nbdoc";
}
