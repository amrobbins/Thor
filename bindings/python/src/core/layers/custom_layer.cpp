#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Optional.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;
using DataType = ThorImplementation::TensorDescriptor::DataType;
using PhysicalTensor = ThorImplementation::Tensor;
using TensorMap = std::unordered_map<std::string, Tensor>;
using PhysicalTensorMap = std::unordered_map<std::string, PhysicalTensor>;
using Expression = ThorImplementation::Expression;
using Outputs = ThorImplementation::Outputs;
using DynamicExpression = ThorImplementation::DynamicExpression;
using DynamicExpressionBuild = ThorImplementation::DynamicExpressionBuild;
using FusedEquation = ThorImplementation::FusedEquation;

namespace {

struct OrderedApiTensorMap {
    std::vector<std::string> names;
    TensorMap tensors;
};

OrderedApiTensorMap apiTensorMapFromPython(nb::dict mapping, const std::string& what) {
    OrderedApiTensorMap result;
    result.names.reserve(mapping.size());

    for (auto item : mapping) {
        if (!nb::isinstance<nb::str>(item.first)) {
            throw std::runtime_error("CustomLayer " + what + " keys must be strings.");
        }
        if (!nb::isinstance<Tensor>(item.second)) {
            throw std::runtime_error("CustomLayer " + what + " values must be thor.Tensor objects.");
        }

        std::string name = nb::cast<std::string>(item.first);
        Tensor tensor = nb::cast<Tensor>(item.second);
        if (result.tensors.contains(name)) {
            throw std::runtime_error("CustomLayer " + what + " contains duplicate name '" + name + "'.");
        }
        result.names.push_back(name);
        result.tensors.emplace(std::move(name), std::move(tensor));
    }

    if (result.names.empty()) {
        throw std::runtime_error("CustomLayer requires at least one " + what + ".");
    }
    return result;
}

std::vector<std::shared_ptr<Parameter>> parametersFromPython(nb::object obj) {
    std::vector<std::shared_ptr<Parameter>> result;
    if (obj.is_none()) {
        return result;
    }

    if (nb::isinstance<nb::dict>(obj)) {
        nb::dict mapping = nb::cast<nb::dict>(obj);
        result.reserve(mapping.size());
        for (auto item : mapping) {
            (void)item.first;
            result.push_back(nb::cast<std::shared_ptr<Parameter>>(item.second));
        }
        return result;
    }

    nb::iterable iterable = nb::cast<nb::iterable>(obj);
    for (nb::handle item : iterable) {
        result.push_back(nb::cast<std::shared_ptr<Parameter>>(item));
    }
    return result;
}

std::vector<std::string> parameterNames(const std::vector<std::shared_ptr<Parameter>>& parameters) {
    std::vector<std::string> names;
    names.reserve(parameters.size());
    std::set<std::string> seen;
    for (const auto& parameter : parameters) {
        if (parameter == nullptr) {
            throw std::runtime_error("CustomLayer received a null Parameter.");
        }
        const std::string& name = parameter->getName();
        if (!seen.insert(name).second) {
            throw std::runtime_error("CustomLayer received duplicate Parameter name '" + name + "'.");
        }
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> concatenateInputNames(const std::vector<std::string>& featureInputNames,
                                               const std::vector<std::string>& parameterNames) {
    std::vector<std::string> result;
    result.reserve(featureInputNames.size() + parameterNames.size());
    std::set<std::string> seen;
    for (const std::string& name : featureInputNames) {
        if (!seen.insert(name).second) {
            throw std::runtime_error("CustomLayer duplicate input name '" + name + "'.");
        }
        result.push_back(name);
    }
    for (const std::string& name : parameterNames) {
        if (!seen.insert(name).second) {
            throw std::runtime_error("CustomLayer Parameter name '" + name + "' conflicts with a feature input name.");
        }
        result.push_back(name);
    }
    return result;
}

std::vector<std::pair<std::string, Expression>> expressionsFromPythonDict(nb::dict mapping) {
    std::vector<std::pair<std::string, Expression>> namedExpressions;
    namedExpressions.reserve(mapping.size());
    for (auto item : mapping) {
        if (!nb::isinstance<nb::str>(item.first)) {
            throw std::runtime_error("CustomLayer build result keys must be strings.");
        }
        if (!nb::isinstance<Expression>(item.second)) {
            throw std::runtime_error("CustomLayer build result values must be thor.physical.Expression objects.");
        }
        namedExpressions.emplace_back(nb::cast<std::string>(item.first), nb::cast<Expression>(item.second));
    }
    return namedExpressions;
}

class CustomLayerApiContext {
   public:
    CustomLayerApiContext(TensorMap inputs, TensorMap outputs) : inputs(std::move(inputs)), outputs(std::move(outputs)) {}

    const TensorMap& getInputs() const { return inputs; }
    const TensorMap& getOutputs() const { return outputs; }

    Tensor input(const std::string& name) const { return getFrom(inputs, name, "input"); }
    Tensor output(const std::string& name) const { return getFrom(outputs, name, "output"); }

    bool hasInput(const std::string& name) const { return inputs.contains(name); }
    bool hasOutput(const std::string& name) const { return outputs.contains(name); }

   private:
    static Tensor getFrom(const TensorMap& tensors, const std::string& name, const std::string& what) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("CustomLayerApiContext has no " + what + " named '" + name + "'.");
        }
        return it->second;
    }

    TensorMap inputs;
    TensorMap outputs;
};

class CustomLayerBuildContext {
   public:
    CustomLayerBuildContext(PhysicalTensorMap allInputs,
                            PhysicalTensorMap outputs,
                            std::vector<std::string> featureInputNames,
                            std::vector<std::string> parameterNames,
                            Stream& stream,
                            bool useFastMath)
        : allInputs(std::move(allInputs)),
          outputs(std::move(outputs)),
          featureInputNames(std::move(featureInputNames)),
          parameterNames(std::move(parameterNames)),
          stream(&stream),
          useFastMath(useFastMath) {
        for (const std::string& name : this->featureInputNames) {
            auto it = this->allInputs.find(name);
            if (it == this->allInputs.end()) {
                throw std::runtime_error("CustomLayerBuildContext missing feature input '" + name + "'.");
            }
            featureInputs.emplace(name, it->second);
        }
        for (const std::string& name : this->parameterNames) {
            auto it = this->allInputs.find(name);
            if (it == this->allInputs.end()) {
                throw std::runtime_error("CustomLayerBuildContext missing parameter tensor '" + name + "'.");
            }
            parameterTensors.emplace(name, it->second);
        }
    }

    const PhysicalTensorMap& inputs() const { return featureInputs; }
    const PhysicalTensorMap& parameters() const { return parameterTensors; }
    const PhysicalTensorMap& all_inputs() const { return allInputs; }
    const PhysicalTensorMap& output_tensors() const { return outputs; }
    Stream& getStream() const { return *stream; }
    int32_t deviceNum() const { return stream->getGpuNum(); }
    bool getUseFastMath() const { return useFastMath; }

    PhysicalTensor inputTensor(const std::string& name) const { return getFrom(featureInputs, name, "feature input"); }
    PhysicalTensor parameterTensor(const std::string& name) const { return getFrom(parameterTensors, name, "parameter"); }
    PhysicalTensor outputTensor(const std::string& name) const { return getFrom(outputs, name, "output"); }

    bool hasInput(const std::string& name) const { return featureInputs.contains(name); }
    bool hasParameter(const std::string& name) const { return parameterTensors.contains(name); }
    bool hasOutput(const std::string& name) const { return outputs.contains(name); }

    Expression input(const std::string& name, nb::object outputDTypeObj, nb::object computeDTypeObj) const {
        require(featureInputs, name, "feature input");
        return expressionInput(name, outputDTypeObj, computeDTypeObj, std::nullopt);
    }

    Expression param(const std::string& name, nb::object outputDTypeObj, nb::object computeDTypeObj) const {
        const PhysicalTensor& tensor = require(parameterTensors, name, "parameter");
        return expressionInput(name, outputDTypeObj, computeDTypeObj, tensor.getDescriptor().getDataType());
    }

   private:
    static PhysicalTensor getFrom(const PhysicalTensorMap& tensors, const std::string& name, const std::string& what) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("CustomLayerBuildContext has no " + what + " named '" + name + "'.");
        }
        return it->second;
    }

    static const PhysicalTensor& require(const PhysicalTensorMap& tensors, const std::string& name, const std::string& what) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("CustomLayerBuildContext has no " + what + " named '" + name + "'.");
        }
        return it->second;
    }

    static Expression expressionInput(const std::string& name,
                                      nb::object outputDTypeObj,
                                      nb::object computeDTypeObj,
                                      std::optional<DataType> defaultDType) {
        Optional<DataType> outputDType = Optional<DataType>::empty();
        Optional<DataType> computeDType = Optional<DataType>::empty();

        if (!outputDTypeObj.is_none()) {
            outputDType = nb::cast<DataType>(outputDTypeObj);
        } else if (defaultDType.has_value()) {
            outputDType = defaultDType.value();
        }

        if (!computeDTypeObj.is_none()) {
            computeDType = nb::cast<DataType>(computeDTypeObj);
        } else if (defaultDType.has_value()) {
            computeDType = defaultDType.value();
        }

        return Expression::input(name, computeDType, outputDType);
    }

    PhysicalTensorMap allInputs;
    PhysicalTensorMap featureInputs;
    PhysicalTensorMap parameterTensors;
    PhysicalTensorMap outputs;
    std::vector<std::string> featureInputNames;
    std::vector<std::string> parameterNames;
    Stream* stream;
    bool useFastMath;
};

DynamicExpression makeDynamicExpressionFromOwner(nb::object buildOwner,
                                                 std::vector<std::string> featureInputNames,
                                                 std::vector<std::string> outputNames,
                                                 std::vector<std::string> parameterNames,
                                                 bool useFastMath) {
    std::vector<std::string> expectedInputNames = concatenateInputNames(featureInputNames, parameterNames);

    if (!nb::hasattr(buildOwner, "build")) {
        throw std::runtime_error("CustomLayer owner object must define build(ctx).");
    }

    return DynamicExpression(
        expectedInputNames,
        outputNames,
        [buildOwner = std::move(buildOwner),
         featureInputNames = std::move(featureInputNames),
         parameterNames = std::move(parameterNames),
         useFastMath](const PhysicalTensorMap& inputs, const PhysicalTensorMap& outputs, Stream& stream) -> DynamicExpressionBuild {
            nb::gil_scoped_acquire gil;

            CustomLayerBuildContext ctx(inputs, outputs, featureInputNames, parameterNames, stream, useFastMath);
            nb::object result = buildOwner.attr("build")(nb::cast(ctx));

            if (nb::isinstance<DynamicExpressionBuild>(result)) {
                return nb::cast<DynamicExpressionBuild>(result);
            }

            if (!nb::isinstance<nb::dict>(result)) {
                throw std::runtime_error(
                    "CustomLayer build(ctx) must return dict[str, thor.physical.Expression] or thor.physical.DynamicExpressionBuild.");
            }

            Outputs expressionOutputs = Expression::outputs(expressionsFromPythonDict(nb::cast<nb::dict>(result)));
            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(
                    FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum(), useFastMath)),
                inputs,
                {},
                outputs,
                {},
            };
        });
}

DynamicExpression makeDynamicExpression(nb::callable buildCallable,
                                        std::vector<std::string> featureInputNames,
                                        std::vector<std::string> outputNames,
                                        std::vector<std::string> parameterNames,
                                        bool useFastMath) {
    std::vector<std::string> expectedInputNames = concatenateInputNames(featureInputNames, parameterNames);

    return DynamicExpression(
        expectedInputNames,
        outputNames,
        [buildCallable = nb::borrow<nb::callable>(buildCallable),
         featureInputNames = std::move(featureInputNames),
         parameterNames = std::move(parameterNames),
         useFastMath](const PhysicalTensorMap& inputs, const PhysicalTensorMap& outputs, Stream& stream) -> DynamicExpressionBuild {
            nb::gil_scoped_acquire gil;

            CustomLayerBuildContext ctx(inputs, outputs, featureInputNames, parameterNames, stream, useFastMath);
            nb::object result = buildCallable(nb::cast(ctx));

            if (nb::isinstance<DynamicExpressionBuild>(result)) {
                return nb::cast<DynamicExpressionBuild>(result);
            }

            if (!nb::isinstance<nb::dict>(result)) {
                throw std::runtime_error(
                    "CustomLayer build(ctx) must return dict[str, thor.physical.Expression] or thor.physical.DynamicExpressionBuild.");
            }

            Outputs expressionOutputs = Expression::outputs(expressionsFromPythonDict(nb::cast<nb::dict>(result)));
            return DynamicExpressionBuild{
                std::make_shared<FusedEquation>(
                    FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum(), useFastMath)),
                inputs,
                {},
                outputs,
                {},
            };
        });
}

class PythonCustomLayerRecipe {
   public:
    virtual ~PythonCustomLayerRecipe() = default;

    virtual nb::object parameters(const CustomLayerApiContext& ctx) {
        (void)ctx;
        return nb::list();
    }

    virtual nb::object build(const CustomLayerBuildContext& ctx) {
        (void)ctx;
        throw std::runtime_error("CustomLayer subclasses must override build(ctx).");
    }

    void initialize(nb::object owner,
                    Network& network,
                    nb::dict inputsObj,
                    nb::dict outputsObj,
                    std::shared_ptr<Optimizer> optimizer,
                    bool inferenceOnly,
                    bool useFastMath) {
        (void)inferenceOnly;

        OrderedApiTensorMap inputs = apiTensorMapFromPython(inputsObj, "inputs");
        OrderedApiTensorMap outputs = apiTensorMapFromPython(outputsObj, "outputs");

        apiInputs = inputs.tensors;
        apiOutputs = outputs.tensors;
        inputNames = inputs.names;
        outputNames = outputs.names;

        CustomLayerApiContext apiCtx(apiInputs, apiOutputs);
        nb::object parametersObj = owner.attr("parameters")(nb::cast(apiCtx));

        std::vector<std::shared_ptr<Parameter>> parameters = parametersFromPython(parametersObj);
        std::vector<std::string> paramNames = parameterNames(parameters);
        parameters_ = parameters;

        DynamicExpression expr = makeDynamicExpressionFromOwner(std::move(owner), inputNames, outputNames, paramNames, useFastMath);

        CustomLayer::Builder builder;
        builder.network(network)
            .expression(std::move(expr))
            .inputNames(inputNames)
            .outputNames(outputNames)
            .inputInterface(apiInputs)
            .outputInterface(apiOutputs)
            .parameters(parameters)
            .useFastMath(useFastMath);

        if (optimizer != nullptr) {
            builder.optimizer(std::move(optimizer));
        }

        CustomLayer built = builder.build();

        // Important ownership model:
        // - The native/network CustomLayer owns this Python recipe through the DynamicExpression closure.
        // - This Python recipe must not strongly own the native CustomLayer back, or we create self -> native -> self.
        // - We only cache the logical output interface here for convenient __getitem__ access.
        outputInterface = built.getOutputInterface(apiInputs);
    }

    Tensor getOutput(const std::string& name) const {
        auto it = outputInterface.find(name);
        if (it == outputInterface.end()) {
            throw std::runtime_error("CustomLayer has no output named '" + name + "'.");
        }
        return it->second;
    }

    TensorMap getOutputInterface() const { return outputInterface; }

    const std::vector<std::string>& getInputNames() const { return inputNames; }
    const std::vector<std::string>& getOutputNames() const { return outputNames; }
    const std::vector<std::shared_ptr<Parameter>>& getParameters() const { return parameters_; }

   private:
    TensorMap apiInputs;
    TensorMap apiOutputs;
    TensorMap outputInterface;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::shared_ptr<Parameter>> parameters_;
};

class PyPythonCustomLayerRecipe : public PythonCustomLayerRecipe {
   public:
    NB_TRAMPOLINE(PythonCustomLayerRecipe, 2);

    nb::object parameters(const CustomLayerApiContext& ctx) override { NB_OVERRIDE(parameters, ctx); }

    nb::object build(const CustomLayerBuildContext& ctx) override { NB_OVERRIDE(build, ctx); }
};

}  // namespace

void bind_custom_layer(nb::module_& layers) {
    auto api_ctx = nb::class_<CustomLayerApiContext>(layers, "CustomLayerApiContext");
    api_ctx.attr("__module__") = "thor.layers";
    api_ctx.def_prop_ro("inputs", &CustomLayerApiContext::getInputs, nb::rv_policy::reference_internal);
    api_ctx.def_prop_ro("outputs", &CustomLayerApiContext::getOutputs, nb::rv_policy::reference_internal);
    api_ctx.def("input", &CustomLayerApiContext::input, "name"_a);
    api_ctx.def("output", &CustomLayerApiContext::output, "name"_a);
    api_ctx.def("has_input", &CustomLayerApiContext::hasInput, "name"_a);
    api_ctx.def("has_output", &CustomLayerApiContext::hasOutput, "name"_a);

    auto ctx = nb::class_<CustomLayerBuildContext>(layers, "CustomLayerBuildContext");
    ctx.attr("__module__") = "thor.layers";
    ctx.def_prop_ro("inputs", &CustomLayerBuildContext::inputs, nb::rv_policy::reference_internal);
    ctx.def_prop_ro("parameters", &CustomLayerBuildContext::parameters, nb::rv_policy::reference_internal);
    ctx.def_prop_ro("all_inputs", &CustomLayerBuildContext::all_inputs, nb::rv_policy::reference_internal);
    ctx.def_prop_ro("outputs", &CustomLayerBuildContext::output_tensors, nb::rv_policy::reference_internal);
    ctx.def_prop_ro("stream", &CustomLayerBuildContext::getStream, nb::rv_policy::reference_internal);
    ctx.def_prop_ro("device_num", &CustomLayerBuildContext::deviceNum);
    ctx.def_prop_ro("use_fast_math", &CustomLayerBuildContext::getUseFastMath);
    ctx.def("input_tensor", &CustomLayerBuildContext::inputTensor, "name"_a);
    ctx.def("parameter_tensor", &CustomLayerBuildContext::parameterTensor, "name"_a);
    ctx.def("param_tensor", &CustomLayerBuildContext::parameterTensor, "name"_a);
    ctx.def("output_tensor", &CustomLayerBuildContext::outputTensor, "name"_a);
    ctx.def("has_input", &CustomLayerBuildContext::hasInput, "name"_a);
    ctx.def("has_parameter", &CustomLayerBuildContext::hasParameter, "name"_a);
    ctx.def("has_param", &CustomLayerBuildContext::hasParameter, "name"_a);
    ctx.def("has_output", &CustomLayerBuildContext::hasOutput, "name"_a);
    ctx.def(
        "input", &CustomLayerBuildContext::input, "name"_a, "output_dtype"_a.none() = nb::none(), "compute_dtype"_a.none() = nb::none());
    ctx.def(
        "param", &CustomLayerBuildContext::param, "name"_a, "output_dtype"_a.none() = nb::none(), "compute_dtype"_a.none() = nb::none());

    auto public_custom_layer = nb::class_<PythonCustomLayerRecipe, PyPythonCustomLayerRecipe>(layers, "CustomLayer");
    public_custom_layer.attr("__module__") = "thor.layers";

    public_custom_layer.def(
        "__init__",
        [](PythonCustomLayerRecipe* self,
           Network& network,
           nb::dict inputsObj,
           nb::dict outputsObj,
           std::shared_ptr<Optimizer> optimizer,
           bool inferenceOnly,
           bool useFastMath) {
            new (self) PythonCustomLayerRecipe();

            // This Python object is the custom layer recipe. The network/native layer owns it
            // through the DynamicExpression closure so users do not need to keep a separate
            // Python reference to the layer object.
            //
            // The recipe intentionally does not own the native CustomLayer back.
            nb::object owner = nb::cast(self, nb::rv_policy::reference);

            self->initialize(std::move(owner), network, inputsObj, outputsObj, std::move(optimizer), inferenceOnly, useFastMath);
        },
        "network"_a,
        "inputs"_a,
        "outputs"_a,
        "optimizer"_a.none() = nb::none(),
        "inference_only"_a = false,
        "use_fast_math"_a = false,
        R"nbdoc(
Subclassable custom trainable layer.

Subclasses normally override:

    parameters(api_ctx) -> list[thor.Parameter] | dict[str, thor.Parameter]
    build(ctx) -> dict[str, thor.physical.Expression] | thor.physical.DynamicExpressionBuild

The public Python object is a recipe. Construction creates a native Thor::CustomLayer
and adds it to the network. The native/network layer owns this recipe for delayed
physical build(ctx) calls, while the recipe does not own the native layer.
        )nbdoc");

    public_custom_layer.def("parameters", &PythonCustomLayerRecipe::parameters, "ctx"_a);
    public_custom_layer.def("build", &PythonCustomLayerRecipe::build, "ctx"_a);
    public_custom_layer.def("get_output", &PythonCustomLayerRecipe::getOutput, "name"_a);
    public_custom_layer.def("__getitem__", &PythonCustomLayerRecipe::getOutput, "name"_a);
    public_custom_layer.def_prop_ro("outputs", &PythonCustomLayerRecipe::getOutputInterface);
    public_custom_layer.def("get_input_names", &PythonCustomLayerRecipe::getInputNames);
    public_custom_layer.def("get_output_names", &PythonCustomLayerRecipe::getOutputNames);
    public_custom_layer.def("get_parameters", &PythonCustomLayerRecipe::getParameters, nb::rv_policy::reference_internal);

    auto custom_layer = nb::class_<CustomLayer, TrainableLayer>(layers, "_CustomLayer");
    custom_layer.attr("__module__") = "thor.layers";

    custom_layer.def(
        "__init__",
        [](CustomLayer* self,
           Network& network,
           nb::dict inputsObj,
           nb::dict outputsObj,
           nb::callable buildCallable,
           nb::object parametersObj,
           std::shared_ptr<Optimizer> optimizer,
           bool inferenceOnly,
           bool useFastMath) {
            (void)inferenceOnly;

            OrderedApiTensorMap inputs = apiTensorMapFromPython(inputsObj, "inputs");
            OrderedApiTensorMap outputs = apiTensorMapFromPython(outputsObj, "outputs");
            std::vector<std::shared_ptr<Parameter>> parameters = parametersFromPython(parametersObj);
            std::vector<std::string> paramNames = parameterNames(parameters);

            DynamicExpression expr = makeDynamicExpression(buildCallable, inputs.names, outputs.names, paramNames, useFastMath);

            CustomLayer::Builder builder;
            builder.network(network)
                .expression(std::move(expr))
                .inputNames(inputs.names)
                .outputNames(outputs.names)
                .inputInterface(inputs.tensors)
                .outputInterface(outputs.tensors)
                .parameters(parameters)
                .useFastMath(useFastMath);

            if (optimizer != nullptr) {
                builder.optimizer(std::move(optimizer));
            }

            CustomLayer built = builder.build();
            new (self) CustomLayer(std::move(built));
        },
        "network"_a,
        "inputs"_a,
        "outputs"_a,
        "build"_a,
        "parameters"_a.none() = nb::none(),
        "optimizer"_a.none() = nb::none(),
        "inference_only"_a = false,
        "use_fast_math"_a = false,
        R"nbdoc(
Internal/native CustomLayer constructor. Most Python users should subclass
thor.layers.CustomLayer instead.
        )nbdoc");

    custom_layer.def("get_output_interface", &CustomLayer::getOutputInterface, "inputs"_a);
    custom_layer.def("get_input_names", &CustomLayer::getInputNames);
    custom_layer.def("get_output_names", &CustomLayer::getOutputNames);
    custom_layer.def("get_parameters", &CustomLayer::getParameters, nb::rv_policy::reference_internal);
}
