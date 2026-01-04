#include "DeepLearning/Api/Initializers/UniformRandom.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json UniformRandom::serialize(Stream stream) const {
    json j;
    j["initializer_type"] = string("uniform_random");
    j["version"] = getVersion();
    j["min_value"] = minValue;
    j["max_value"] = maxValue;
    return j;
}

shared_ptr<Initializer> UniformRandom::deserialize(const json &j) {
    if (j.at("initializer_type").get<string>() != "uniform_random")
        throw runtime_error("Layer type mismatch in UniformRandom::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in UniformRandom::deserialize: " + j["version"].get<string>());

    UniformRandom uniformRandom;
    uniformRandom.minValue = j.at("min_value").get<double>();
    uniformRandom.maxValue = j.at("max_value").get<double>();
    uniformRandom.initialized = true;
    return uniformRandom.clone();
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Initializer::registerLayer("uniform_random", &Thor::UniformRandom::deserialize);
    return true;
}();
}  // namespace
