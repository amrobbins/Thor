#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
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

class GilSafePythonObject {
   public:
    explicit GilSafePythonObject(nb::handle object) : object(object.ptr()) {
        nb::gil_scoped_acquire gil;
        Py_XINCREF(this->object);
    }

    GilSafePythonObject(const GilSafePythonObject&) = delete;
    GilSafePythonObject& operator=(const GilSafePythonObject&) = delete;

    ~GilSafePythonObject() {
        if (object == nullptr)
            return;

        nb::gil_scoped_acquire gil;
        Py_XDECREF(object);
    }

    nb::handle get() const { return nb::handle(object); }

   private:
    PyObject* object = nullptr;
};

struct OrderedApiTensorMap {
    std::vector<std::string> names;
    TensorMap tensors;
};

OrderedApiTensorMap apiTensorMapFromPythonDict(nb::dict mapping, const std::string& what) {
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

OrderedApiTensorMap normalizeInputs(nb::object inputsObj) {
    if (nb::isinstance<Tensor>(inputsObj)) {
        OrderedApiTensorMap result;
        result.names.push_back("feature_input");
        result.tensors.emplace("feature_input", nb::cast<Tensor>(inputsObj));
        return result;
    }

    if (!nb::isinstance<nb::dict>(inputsObj)) {
        throw std::runtime_error("CustomLayer inputs must be a thor.Tensor or a mapping of input name to thor.Tensor.");
    }

    return apiTensorMapFromPythonDict(nb::cast<nb::dict>(inputsObj), "inputs");
}

std::vector<std::string> normalizeOutputNames(nb::object outputNamesObj, const OrderedApiTensorMap& inputs) {
    if (outputNamesObj.is_none()) {
        if (inputs.names.size() == 1 && inputs.names[0] == "feature_input") {
            return {"feature_output"};
        }
        throw std::runtime_error(
            "CustomLayer with named inputs requires output_names. Omit output_names only when using the single-tensor convenience form.");
    }

    if (nb::isinstance<nb::str>(outputNamesObj)) {
        return {nb::cast<std::string>(outputNamesObj)};
    }

    if (!nb::isinstance<nb::sequence>(outputNamesObj)) {
        throw std::runtime_error("CustomLayer output_names must be a string or a sequence of strings.");
    }

    std::vector<std::string> names;
    nb::sequence sequence = nb::cast<nb::sequence>(outputNamesObj);
    for (nb::handle item : sequence) {
        if (!nb::isinstance<nb::str>(item)) {
            throw std::runtime_error("CustomLayer output names must be strings.");
        }
        names.push_back(nb::cast<std::string>(item));
    }

    if (names.empty()) {
        throw std::runtime_error("CustomLayer requires at least one output name.");
    }

    return names;
}

std::vector<std::shared_ptr<ParameterSpecification>> parametersFromPythonObject(nb::object obj) {
    std::vector<std::shared_ptr<ParameterSpecification>> result;
    if (obj.is_none()) {
        return result;
    }

    if (nb::isinstance<nb::dict>(obj)) {
        throw std::runtime_error(
            "CustomLayer parameters() must return list[thor.ParameterSpecification], not dict[str, thor.ParameterSpecification]. "
            "Parameter names are owned by the Parameter objects themselves.");
    }

    if (!nb::isinstance<nb::list>(obj)) {
        throw std::runtime_error("CustomLayer parameters() must return list[thor.ParameterSpecification].");
    }

    nb::list parameters = nb::cast<nb::list>(obj);
    result.reserve(parameters.size());
    for (nb::handle item : parameters) {
        result.push_back(nb::cast<std::shared_ptr<ParameterSpecification>>(item));
    }
    return result;
}

nb::callable callableFromPythonObject(nb::object obj, const std::string& what) {
    if (!nb::isinstance<nb::callable>(obj)) {
        throw std::runtime_error("CustomLayer " + what + " must be callable.");
    }

    return nb::borrow<nb::callable>(obj);
}

std::vector<std::string> parameterNames(const std::vector<std::shared_ptr<ParameterSpecification>>& parameters) {
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

std::set<std::string> toNameSet(const std::vector<std::string>& names) { return std::set<std::string>(names.begin(), names.end()); }

std::set<std::string> toNameSet(const PhysicalTensorMap& tensors) {
    std::set<std::string> names;
    for (const auto& [name, _] : tensors) {
        names.insert(name);
    }
    return names;
}

std::set<std::string> toNameSet(const std::vector<ThorImplementation::NamedOutput>& outputs) {
    std::set<std::string> names;
    for (const auto& output : outputs) {
        names.insert(output.name);
    }
    return names;
}

void validateCustomLayerExpressionInputs(const std::vector<std::string>& expectedInputNames,
                                         const std::set<std::string>& actualInputNames) {
    const std::set<std::string> expectedInputNameSet = toNameSet(expectedInputNames);

    std::set<std::string> unknownActualInputs;
    for (const auto& name : actualInputNames) {
        if (!expectedInputNameSet.contains(name)) {
            unknownActualInputs.insert(name);
        }
    }

    if (unknownActualInputs.empty()) {
        return;
    }

    std::string expectedInputNamesString;
    for (const auto& name : expectedInputNameSet) {
        expectedInputNamesString += name + " ";
    }

    std::string actualInputNamesString;
    for (const auto& name : actualInputNames) {
        actualInputNamesString += name + " ";
    }

    std::string unknownActualInputsString;
    for (const auto& name : unknownActualInputs) {
        unknownActualInputsString += name + " ";
    }

    throw std::runtime_error("CustomLayer expression used undeclared inputs. Declared inputs: " + expectedInputNamesString +
                             " Actual inputs used by prepared expression: " + actualInputNamesString +
                             " Unknown inputs: " + unknownActualInputsString);
}

PhysicalTensorMap selectNamedTensors(const PhysicalTensorMap& tensors, const std::set<std::string>& names, const std::string& what) {
    PhysicalTensorMap selected;
    for (const std::string& name : names) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("CustomLayer missing " + what + " '" + name + "'.");
        }
        selected.emplace(name, it->second);
    }
    return selected;
}

void validateCustomLayerForwardOutputs(const std::vector<std::string>& expectedOutputNames,
                                       const std::set<std::string>& actualOutputNames) {
    const std::set<std::string> expectedOutputNameSet = toNameSet(expectedOutputNames);
    if (actualOutputNames == expectedOutputNameSet) {
        return;
    }

    std::string expectedOutputNamesString;
    for (const auto& name : expectedOutputNameSet) {
        expectedOutputNamesString += name + " ";
    }

    std::string actualOutputNamesString;
    for (const auto& name : actualOutputNames) {
        actualOutputNamesString += name + " ";
    }

    throw std::runtime_error("CustomLayer forward output mismatch. Expected outputs: " + expectedOutputNamesString +
                             " Actual outputs returned by build(context): " + actualOutputNamesString);
}

void validateDeclaredParametersReferenced(const PhysicalTensorMap& parameterTensors,
                                          const std::set<std::string>& actualInputNames,
                                          const std::vector<std::string>& outputNames) {
    const std::set<std::string> outputNameSet = toNameSet(outputNames);

    for (const auto& [name, _] : parameterTensors) {
        // Defer parameter/output name collisions to the implementation-side placement path.
        if (outputNameSet.contains(name)) {
            continue;
        }

        if (!actualInputNames.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }
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

PhysicalTensorMap selectNamedTensors(const PhysicalTensorMap& tensors, const std::vector<std::string>& names, const std::string& what) {
    PhysicalTensorMap selected;
    for (const std::string& name : names) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("CustomLayer missing " + what + " '" + name + "'.");
        }
        selected.emplace(name, it->second);
    }
    return selected;
}

class CustomLayerBuildContext {
   public:
    CustomLayerBuildContext(
        PhysicalTensorMap featureInputs, PhysicalTensorMap parameterTensors, PhysicalTensorMap outputs, Stream& stream, bool useFastMath)
        : featureInputs(std::move(featureInputs)),
          parameterTensors(std::move(parameterTensors)),
          outputs(std::move(outputs)),
          stream(&stream),
          useFastMath(useFastMath) {}

    const PhysicalTensorMap& inputs() const { return featureInputs; }
    const PhysicalTensorMap& parameters() const { return parameterTensors; }
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

    PhysicalTensorMap featureInputs;
    PhysicalTensorMap parameterTensors;
    PhysicalTensorMap outputs;
    Stream* stream;
    bool useFastMath;
};

DynamicExpressionBuild callBuildCallableForContext(nb::callable callable,
                                                   const std::vector<std::string>& expectedInputNames,
                                                   const std::vector<std::string>& outputNames,
                                                   const PhysicalTensorMap& inputs,
                                                   const PhysicalTensorMap& featureInputs,
                                                   const PhysicalTensorMap& parameterTensors,
                                                   const PhysicalTensorMap& outputs,
                                                   Stream& stream,
                                                   bool useFastMath) {
    CustomLayerBuildContext context(featureInputs, parameterTensors, outputs, stream, useFastMath);
    nb::object result = callable(nb::cast(context));

    if (nb::isinstance<DynamicExpressionBuild>(result)) {
        DynamicExpressionBuild build = nb::cast<DynamicExpressionBuild>(result);
        validateCustomLayerExpressionInputs(expectedInputNames, toNameSet(build.stamp_inputs));
        validateCustomLayerForwardOutputs(outputNames, toNameSet(build.equation->getOutputNames()));
        return build;
    }

    if (!nb::isinstance<nb::dict>(result)) {
        throw std::runtime_error(
            "CustomLayer build(context) must return dict[str, thor.physical.Expression] or thor.physical.DynamicExpressionBuild.");
    }

    Outputs expressionOutputs = Expression::outputs(expressionsFromPythonDict(nb::cast<nb::dict>(result)));
    std::set<std::string> actualInputNames = expressionOutputs.expression()->getInputNames();
    validateCustomLayerExpressionInputs(expectedInputNames, actualInputNames);
    validateDeclaredParametersReferenced(parameterTensors, actualInputNames, outputNames);
    validateCustomLayerForwardOutputs(outputNames, toNameSet(expressionOutputs.namedOutputs()));

    PhysicalTensorMap usedInputs = selectNamedTensors(inputs, actualInputNames, "expression input");

    return DynamicExpressionBuild{
        std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum(), useFastMath)),
        usedInputs,
        {},
        outputs,
        {},
    };
}

DynamicExpression makeDynamicExpressionFromCallable(nb::callable buildCallable,
                                                    std::vector<std::string> featureInputNames,
                                                    std::vector<std::string> outputNames,
                                                    std::vector<std::string> parameterNames,
                                                    bool useFastMath) {
    std::vector<std::string> expectedInputNames = concatenateInputNames(featureInputNames, parameterNames);
    auto buildCallableRef = std::make_shared<GilSafePythonObject>(buildCallable);

    return DynamicExpression(
        expectedInputNames,
        outputNames,
        [buildCallableRef,
         expectedInputNames,
         outputNames = std::move(outputNames),
         featureInputNames = std::move(featureInputNames),
         parameterNames = std::move(parameterNames),
         useFastMath](const PhysicalTensorMap& inputs, const PhysicalTensorMap& outputs, Stream& stream) -> DynamicExpressionBuild {
            nb::gil_scoped_acquire gil;

            PhysicalTensorMap featureInputs = selectNamedTensors(inputs, featureInputNames, "feature input");
            PhysicalTensorMap parameterTensors = selectNamedTensors(inputs, parameterNames, "parameter");
            nb::callable callable = nb::borrow<nb::callable>(buildCallableRef->get());
            return callBuildCallableForContext(
                callable, expectedInputNames, outputNames, inputs, featureInputs, parameterTensors, outputs, stream, useFastMath);
        });
}

DynamicExpression makeDynamicExpressionFromSelf(nb::handle selfHandle,
                                                std::vector<std::string> featureInputNames,
                                                std::vector<std::string> outputNames,
                                                std::vector<std::string> parameterNames,
                                                bool useFastMath) {
    nb::gil_scoped_acquire gil;
    nb::object weakrefModule = nb::module_::import_("weakref");
    nb::object selfWeakref = weakrefModule.attr("ref")(nb::borrow<nb::object>(selfHandle));
    auto selfWeakrefRef = std::make_shared<GilSafePythonObject>(selfWeakref);

    std::vector<std::string> expectedInputNames = concatenateInputNames(featureInputNames, parameterNames);
    return DynamicExpression(
        expectedInputNames,
        outputNames,
        [selfWeakrefRef,
         expectedInputNames,
         outputNames = std::move(outputNames),
         featureInputNames = std::move(featureInputNames),
         parameterNames = std::move(parameterNames),
         useFastMath](const PhysicalTensorMap& inputs, const PhysicalTensorMap& outputs, Stream& stream) -> DynamicExpressionBuild {
            nb::gil_scoped_acquire gil;

            nb::object owner = nb::borrow<nb::object>(selfWeakrefRef->get())();
            if (owner.is_none()) {
                throw std::runtime_error("Python CustomLayer object no longer exists.");
            }

            nb::object buildAttr = owner.attr("build");
            if (!nb::isinstance<nb::callable>(buildAttr)) {
                throw std::runtime_error("CustomLayer build attribute is not callable.");
            }

            PhysicalTensorMap featureInputs = selectNamedTensors(inputs, featureInputNames, "feature input");
            PhysicalTensorMap parameterTensors = selectNamedTensors(inputs, parameterNames, "parameter");
            return callBuildCallableForContext(nb::cast<nb::callable>(buildAttr),
                                               expectedInputNames,
                                               outputNames,
                                               inputs,
                                               featureInputs,
                                               parameterTensors,
                                               outputs,
                                               stream,
                                               useFastMath);
        });
}

}  // namespace

void bind_custom_layer(nb::module_& layers) {
    auto context = nb::class_<CustomLayerBuildContext>(layers, "CustomLayerBuildContext");
    context.attr("__module__") = "thor.layers";
    context.def_prop_ro("inputs", &CustomLayerBuildContext::inputs, nb::rv_policy::reference_internal);
    context.def_prop_ro("input_tensors", &CustomLayerBuildContext::inputs, nb::rv_policy::reference_internal);
    context.def_prop_ro("parameters", &CustomLayerBuildContext::parameters, nb::rv_policy::reference_internal);
    context.def_prop_ro("parameter_tensors", &CustomLayerBuildContext::parameters, nb::rv_policy::reference_internal);
    context.def_prop_ro("param_tensors", &CustomLayerBuildContext::parameters, nb::rv_policy::reference_internal);
    context.def_prop_ro("outputs", &CustomLayerBuildContext::output_tensors, nb::rv_policy::reference_internal);
    context.def_prop_ro("output_tensors", &CustomLayerBuildContext::output_tensors, nb::rv_policy::reference_internal);
    context.def_prop_ro("stream", &CustomLayerBuildContext::getStream, nb::rv_policy::reference_internal);
    context.def_prop_ro("device_num", &CustomLayerBuildContext::deviceNum);
    context.def_prop_ro("use_fast_math", &CustomLayerBuildContext::getUseFastMath);
    context.def("input_tensor", &CustomLayerBuildContext::inputTensor, "name"_a);
    context.def("parameter_tensor", &CustomLayerBuildContext::parameterTensor, "name"_a);
    context.def("param_tensor", &CustomLayerBuildContext::parameterTensor, "name"_a);
    context.def("output_tensor", &CustomLayerBuildContext::outputTensor, "name"_a);
    context.def("has_input", &CustomLayerBuildContext::hasInput, "name"_a);
    context.def("has_parameter", &CustomLayerBuildContext::hasParameter, "name"_a);
    context.def("has_param", &CustomLayerBuildContext::hasParameter, "name"_a);
    context.def("has_output", &CustomLayerBuildContext::hasOutput, "name"_a);
    context.def(
        "input", &CustomLayerBuildContext::input, "name"_a, "output_dtype"_a.none() = nb::none(), "compute_dtype"_a.none() = nb::none());
    context.def(
        "param", &CustomLayerBuildContext::param, "name"_a, "output_dtype"_a.none() = nb::none(), "compute_dtype"_a.none() = nb::none());

    auto custom_layer = nb::class_<CustomLayer, TrainableLayer>(layers, "CustomLayer", nb::is_weak_referenceable());
    custom_layer.attr("__module__") = "thor.layers";

    custom_layer.def(
        "__init__",
        [](nb::handle pySelf,
           Network& network,
           nb::object inputsObj,
           nb::object outputNamesObj,
           nb::object buildObj,
           nb::object parametersObj,
           std::shared_ptr<Optimizer> optimizer,
           bool useFastMath) {
            OrderedApiTensorMap inputs = normalizeInputs(inputsObj);
            CustomLayer* self = nb::inst_ptr<CustomLayer>(pySelf.ptr());

            std::vector<std::string> outputNames = normalizeOutputNames(outputNamesObj, inputs);

            nb::object pySelfObj;
            if (parametersObj.is_none() || buildObj.is_none()) {
                pySelfObj = nb::borrow<nb::object>(pySelf);
            }

            std::vector<std::shared_ptr<ParameterSpecification>> parameters;
            if (parametersObj.is_none()) {
                nb::gil_scoped_acquire gil;
                if (!pySelfObj.is_valid()) {
                    throw std::runtime_error("CustomLayer constructor could not access the Python instance for parameters().");
                }
                nb::object hookResult = pySelfObj.attr("parameters")();
                parameters = parametersFromPythonObject(hookResult);
            } else {
                parameters = parametersFromPythonObject(parametersObj);
            }
            std::vector<std::string> paramNames = parameterNames(parameters);

            DynamicExpression expr =
                buildObj.is_none()
                    ? makeDynamicExpressionFromSelf(
                          pySelfObj.is_valid() ? nb::handle(pySelfObj) : pySelf, inputs.names, outputNames, paramNames, useFastMath)
                    : makeDynamicExpressionFromCallable(
                          callableFromPythonObject(buildObj, "build"), inputs.names, outputNames, paramNames, useFastMath);

            CustomLayer::Builder builder;
            builder.network(network)
                .expression(std::move(expr))
                .inputNames(inputs.names)
                .outputNames(outputNames)
                .inputInterface(inputs.tensors)
                .parameters(parameters)
                .useFastMath(useFastMath);

            if (optimizer != nullptr) {
                builder.optimizer(std::move(optimizer));
            }

            CustomLayer built = builder.build();
            new (self) CustomLayer(std::move(built));
            nb::inst_mark_ready(pySelf);
        },
        "network"_a,
        "inputs"_a,
        "output_names"_a.none() = nb::none(),
        "build"_a.none() = nb::none(),
        "parameters"_a.none() = nb::none(),
        "optimizer"_a.none() = nb::none(),
        "use_fast_math"_a = false,
        R"nbdoc(
Python-facing CustomLayer.

The C++ API layer owns the CustomLayer construction logic, including named input/output
interfaces and logical output tensor inference. Python can use it either directly by
passing build=... and parameters=..., or by subclassing and overriding parameters()
and build(context).

Convenience forms:
- inputs=<thor.Tensor> defaults to {"feature_input": tensor}
- output_names omitted defaults to ["feature_output"]
        )nbdoc");

    custom_layer.def("parameters", [](nb::handle) { return nb::list(); });
    custom_layer.def(
        "build",
        [](nb::handle, const CustomLayerBuildContext&) -> nb::dict {
            throw std::runtime_error("CustomLayer subclasses must override build(context), or pass build=... to the constructor.");
        },
        "context"_a);

    custom_layer.def("get_input_interface", &CustomLayer::getInputInterface, "interface_index"_a = 0);
    custom_layer.def("get_output_interface", nb::overload_cast<const TensorMap&>(&CustomLayer::getOutputInterface, nb::const_), "inputs"_a);
    custom_layer.def("get_output_interface_by_index", &CustomLayer::getOutputInterfaceByIndex, "interface_index"_a = 0);
    custom_layer.def("get_output", &CustomLayer::getOutput, "name"_a, "interface_index"_a = 0);
    custom_layer.def("__getitem__", [](const CustomLayer& self, const std::string& name) { return self.getOutput(name); }, "name"_a);
    custom_layer.def_prop_ro("outputs", [](const CustomLayer& self) { return self.getOutputInterfaceByIndex(0); });
    custom_layer.def("get_input_names", &CustomLayer::getInputNames);
    custom_layer.def("get_output_names", &CustomLayer::getOutputNames);
    custom_layer.def("get_parameters", &CustomLayer::getParameters, nb::rv_policy::reference_internal);
    custom_layer.def("get_bound_parameter", &CustomLayer::getBoundParameter, "placed_network"_a, "name"_a);
    custom_layer.def("get_bound_parameters", &CustomLayer::getBoundParameters, "placed_network"_a);
}
