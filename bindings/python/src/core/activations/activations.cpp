#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <utility>
#include <vector>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/Expression.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

namespace {
using DataType = ThorImplementation::DataType;

std::optional<DataType> optionalDataTypeFromPython(const nb::object& obj) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    return nb::cast<DataType>(obj);
}

ThorImplementation::Expression makePythonActivationEpilogueInput(const nb::object& outputDTypeObj, const nb::object& computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj);
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj);
    return Activation::epilogueInput(computeDType, outputDType);
}

ThorImplementation::Expression makePythonActivationEpilogueAuxInput(const std::string& inputName,
                                                                    const nb::object& outputDTypeObj,
                                                                    const nb::object& computeDTypeObj) {
    std::optional<DataType> outputDType = optionalDataTypeFromPython(outputDTypeObj);
    std::optional<DataType> computeDType = optionalDataTypeFromPython(computeDTypeObj);
    return Activation::epilogueAuxInput(inputName, computeDType, outputDType);
}

std::vector<std::pair<std::string, Tensor>> activationEpilogueInputsFromPython(const nb::object& epilogueInputs) {
    std::vector<std::pair<std::string, Tensor>> bindings;
    if (epilogueInputs.is_none()) {
        return bindings;
    }
    if (!nb::isinstance<nb::dict>(epilogueInputs)) {
        throw nb::type_error("epilogue_inputs must be a dict[str, thor.Tensor] or None");
    }
    nb::dict inputsDict = nb::cast<nb::dict>(epilogueInputs);
    bindings.reserve(inputsDict.size());
    for (auto item : inputsDict) {
        bindings.emplace_back(nb::cast<std::string>(item.first), nb::cast<Tensor>(item.second));
    }
    return bindings;
}

std::optional<ThorImplementation::Expression> activationEpilogueFromPython(const nb::object& epilogue) {
    if (epilogue.is_none()) {
        return std::nullopt;
    }
    if (!nb::isinstance<ThorImplementation::Expression>(epilogue)) {
        throw nb::type_error("epilogue must be a thor.physical.Expression instance or None");
    }
    return nb::cast<ThorImplementation::Expression>(epilogue);
}
}  // namespace

void bind_glu(nb::module_ &m);
void bind_reglu(nb::module_ &m);
void bind_geglu(nb::module_ &m);
void bind_swiglu(nb::module_ &m);
void bind_bilinear_glu(nb::module_ &m);
void bind_elu(nb::module_ &m);
void bind_exponential(nb::module_ &m);
void bind_gelu(nb::module_ &m);
void bind_hard_sigmoid(nb::module_ &m);
void bind_hard_swish(nb::module_ &m);
void bind_hard_tanh(nb::module_ &m);
void bind_mish(nb::module_ &m);
void bind_relu(nb::module_ &m);
void bind_relu6(nb::module_ &m);
void bind_selu(nb::module_ &m);
void bind_sigmoid(nb::module_ &m);
void bind_soft_plus(nb::module_ &m);
void bind_soft_sign(nb::module_ &m);
void bind_softmax(nb::module_ &m);
void bind_swish(nb::module_ &m);
void bind_tanh(nb::module_ &m);
void bind_threshold(nb::module_ &m);

void bind_activations(nb::module_ &activations) {
    activations.doc() = "Thor activations";

    auto activation = nb::class_<Activation>(activations, "Activation");
    activation.attr("__module__") = "thor.activations";
    activation.def(
        "to_expression",
        [](const Activation& self, const ThorImplementation::Expression& input) { return self.toExpression(input); },
        "input"_a,
        R"nbdoc(Return an expression equivalent to applying this activation to the supplied expression.)nbdoc");

    activation.def_static(
        "epilogue_input",
        &makePythonActivationEpilogueInput,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(Return the primary tensor input expression expected by an activation epilogue.)nbdoc");

    activation.def_static(
        "epilogue_aux_input",
        &makePythonActivationEpilogueAuxInput,
        "name"_a,
        "output_dtype"_a.none() = nb::none(),
        "compute_dtype"_a.none() = nb::none(),
        R"nbdoc(Return a named auxiliary tensor input expression for an activation epilogue.)nbdoc");

    activation.def(
        "add_to_network",
        [](Activation& self, Network& network, Tensor featureInput, const nb::object& epilogue, const nb::object& epilogueInputs) {
            return self.addToNetwork(
                featureInput, &network, activationEpilogueFromPython(epilogue), activationEpilogueInputsFromPython(epilogueInputs));
        },
        "network"_a,
        "feature_input"_a,
        "epilogue"_a.none() = nb::none(),
        "epilogue_inputs"_a.none() = nb::none(),
        R"nbdoc(Attach this activation as a standalone expression-backed layer and return its feature output tensor.)nbdoc");

    bind_glu(activations);
    bind_reglu(activations);
    bind_geglu(activations);
    bind_swiglu(activations);
    bind_bilinear_glu(activations);
    bind_elu(activations);
    bind_exponential(activations);
    bind_gelu(activations);
    bind_hard_sigmoid(activations);
    bind_hard_swish(activations);
    bind_hard_tanh(activations);
    bind_mish(activations);
    bind_relu(activations);
    bind_relu6(activations);
    bind_selu(activations);
    bind_sigmoid(activations);
    bind_soft_plus(activations);
    bind_soft_sign(activations);
    bind_softmax(activations);
    bind_swish(activations);
    bind_tanh(activations);
    bind_threshold(activations);
}
