#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"

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
            throw nb::value_error("RMSNorm instance: feature_input must have at least one feature dimension.");
        }
        return {dims.back()};
    }
    return nb::cast<vector<uint64_t>>(obj);
}

std::optional<DataType> optionalDataTypeFromPython(const nb::object& obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return nb::cast<DataType>(obj);
}

ThorImplementation::Expression makePythonEpilogueInput(const nb::object& outputDTypeObj, const nb::object& computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj);
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj);
    return RMSNorm::epilogueInput(computeDType, outputDType);
}

void applyPythonEpilogue(RMSNorm::Builder& builder, const nb::object& epilogue) {
    if (epilogue.is_none()) {
        return;
    }
    if (!nb::isinstance<ThorImplementation::Expression>(epilogue)) {
        throw nb::type_error("epilogue must be a thor.physical.Expression instance or None");
    }
    builder.epilogue(nb::cast<ThorImplementation::Expression>(epilogue));
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
           nb::object parameter_data_type,
           shared_ptr<Initializer> weights_initializer,
           shared_ptr<Optimizer> weights_optimizer,
           nb::object epilogue) {
            if (!(epsilon > 0.0)) {
                throw nb::value_error("RMSNorm instance: epsilon must be > 0.");
            }

            vector<uint64_t> shape = normalizedShapeFromPython(normalized_shape, feature_input);

            RMSNorm::Builder builder;
            builder.network(network).featureInput(feature_input).normalizedShape(shape).epsilon(epsilon);
            if (!parameter_data_type.is_none())
                builder.parameterDataType(nb::cast<DataType>(parameter_data_type));
            applyPythonEpilogue(builder, epilogue);
            if (weights_initializer != nullptr)
                builder.weightsInitializer(weights_initializer);
            builder.weightsOptimizer(weights_optimizer);

            new (self) RMSNorm(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "normalized_shape"_a.none() = nb::none(),
        "epsilon"_a = 1.0e-5,
        "parameter_data_type"_a.none() = nb::none(),
        "weights_initializer"_a.none() = nb::none(),
        "weights_optimizer"_a.none() = nb::none(),
        "epilogue"_a.none() = nb::none());

    rms_norm.def_static(
        "epilogue_input",
        &makePythonEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(
            Return the single tensor input expression expected by an RMSNorm epilogue.
            )nbdoc");

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
        parameter_data_type : thor.DataType or None, default None
            Data type for scale weights. None chooses fp32.
        epilogue : thor.physical.Expression or None, default None
            Optional expression applied after RMSNorm. Build it from ``RMSNorm.epilogue_input()``.
            A Swish/SiLU epilogue can use the cuDNN Frontend RMSNorm + SiLU inference fusion when the
            feature input, output, and scale weights are bf16.
        )nbdoc";
}
