#include "DeepLearning/Api/Initializers/Glorot.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json Glorot::serialize() const {
    json j;
    j["initializer_type"] = string("glorot");
    j["version"] = getVersion();
    j["mode"] = mode;
    return j;
}

shared_ptr<Initializer> Glorot::deserialize(const json &j) {
    if (j.at("initializer_type").get<string>() != "glorot")
        throw runtime_error("Layer type mismatch in Glorot::deserialize: " + j.at("type").get<string>());
    if (j.at("version").get<string>() != "1.0.0")
        throw runtime_error("Unsupported version in Glorot::deserialize: " + j["version"].get<string>());

    ThorImplementation::Glorot::Mode mode = j.at("mode").get<ThorImplementation::Glorot::Mode>();
    Glorot glorot;
    glorot.mode = mode;
    glorot.initialized = true;
    return glorot.clone();
}

}  // namespace Thor

namespace {
static const bool registered = [] {
    Thor::Initializer::registerLayer("glorot", &Thor::Glorot::deserialize);
    return true;
}();
}  // namespace
