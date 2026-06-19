#include "DeepLearning/Api/Parameter/ParameterConstraint.h"

#include "DeepLearning/Implementation/Parameter/ParameterConstraint.h"

#include <memory>
#include <stdexcept>
#include <string>

using nlohmann::json;
using namespace std;

namespace Thor {
namespace {

void validateMinMax(double minValue, double maxValue) {
    if (minValue > maxValue) {
        throw runtime_error("MinMaxParameterConstraint requires min_value <= max_value.");
    }
}

}  // namespace

json ParameterConstraint::architectureJson() const {
    return json{{"version", getVersion()}, {"constraint_type", getConstraintType()}};
}

string ParameterConstraint::getVersion() { return "1.0.0"; }

shared_ptr<ParameterConstraint> ParameterConstraint::deserialize(const json& j) {
    if (j.at("version").get<string>() != getVersion()) {
        throw runtime_error("Unsupported version in ParameterConstraint::deserialize: " + j.at("version").get<string>());
    }
    const string type = j.at("constraint_type").get<string>();
    if (type == "non_negative") {
        return make_shared<NonNegativeParameterConstraint>();
    }
    if (type == "non_positive") {
        return make_shared<NonPositiveParameterConstraint>();
    }
    if (type == "min") {
        return make_shared<MinParameterConstraint>(j.at("min_value").get<double>());
    }
    if (type == "max") {
        return make_shared<MaxParameterConstraint>(j.at("max_value").get<double>());
    }
    if (type == "min_max") {
        return make_shared<MinMaxParameterConstraint>(j.at("min_value").get<double>(), j.at("max_value").get<double>());
    }
    throw runtime_error("Unknown ParameterConstraint type: " + type + ".");
}

shared_ptr<ParameterConstraint> NonNegativeParameterConstraint::clone() const {
    return make_shared<NonNegativeParameterConstraint>(*this);
}

shared_ptr<ThorImplementation::ParameterConstraint> NonNegativeParameterConstraint::stamp() const {
    return make_shared<ThorImplementation::NonNegativeParameterConstraint>();
}

string NonNegativeParameterConstraint::getConstraintType() const { return "non_negative"; }

shared_ptr<ParameterConstraint> NonPositiveParameterConstraint::clone() const {
    return make_shared<NonPositiveParameterConstraint>(*this);
}

shared_ptr<ThorImplementation::ParameterConstraint> NonPositiveParameterConstraint::stamp() const {
    return make_shared<ThorImplementation::NonPositiveParameterConstraint>();
}

string NonPositiveParameterConstraint::getConstraintType() const { return "non_positive"; }

MinParameterConstraint::MinParameterConstraint(double minValue) : minValue(minValue) {}

double MinParameterConstraint::getMinValue() const { return minValue; }

shared_ptr<ParameterConstraint> MinParameterConstraint::clone() const { return make_shared<MinParameterConstraint>(*this); }

shared_ptr<ThorImplementation::ParameterConstraint> MinParameterConstraint::stamp() const {
    return make_shared<ThorImplementation::MinParameterConstraint>(minValue);
}

string MinParameterConstraint::getConstraintType() const { return "min"; }

json MinParameterConstraint::architectureJson() const {
    json j = ParameterConstraint::architectureJson();
    j["min_value"] = minValue;
    return j;
}

MaxParameterConstraint::MaxParameterConstraint(double maxValue) : maxValue(maxValue) {}

double MaxParameterConstraint::getMaxValue() const { return maxValue; }

shared_ptr<ParameterConstraint> MaxParameterConstraint::clone() const { return make_shared<MaxParameterConstraint>(*this); }

shared_ptr<ThorImplementation::ParameterConstraint> MaxParameterConstraint::stamp() const {
    return make_shared<ThorImplementation::MaxParameterConstraint>(maxValue);
}

string MaxParameterConstraint::getConstraintType() const { return "max"; }

json MaxParameterConstraint::architectureJson() const {
    json j = ParameterConstraint::architectureJson();
    j["max_value"] = maxValue;
    return j;
}

MinMaxParameterConstraint::MinMaxParameterConstraint(double minValue, double maxValue) : minValue(minValue), maxValue(maxValue) {
    validateMinMax(minValue, maxValue);
}

double MinMaxParameterConstraint::getMinValue() const { return minValue; }

double MinMaxParameterConstraint::getMaxValue() const { return maxValue; }

shared_ptr<ParameterConstraint> MinMaxParameterConstraint::clone() const { return make_shared<MinMaxParameterConstraint>(*this); }

shared_ptr<ThorImplementation::ParameterConstraint> MinMaxParameterConstraint::stamp() const {
    return make_shared<ThorImplementation::MinMaxParameterConstraint>(minValue, maxValue);
}

string MinMaxParameterConstraint::getConstraintType() const { return "min_max"; }

json MinMaxParameterConstraint::architectureJson() const {
    json j = ParameterConstraint::architectureJson();
    j["min_value"] = minValue;
    j["max_value"] = maxValue;
    return j;
}

}  // namespace Thor
