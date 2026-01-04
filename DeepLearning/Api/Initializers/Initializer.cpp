#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/Network.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

string Initializer::getVersion() const { return "1.0.0"; }

unordered_map<string, Initializer::Deserializer> &Initializer::getRegistry() {
    static unordered_map<string, Deserializer> registry;
    return registry;
}

void Initializer::registerLayer(string name, Deserializer fn) { getRegistry().emplace(std::move(name), std::move(fn)); }

shared_ptr<Initializer> Initializer::deserialize(const json &j) {
    assert(j.contains("initializer_type"));
    string initializerType = j.at("initializer_type").get<string>();

    unordered_map<string, Deserializer> &registry = getRegistry();
    auto it = registry.find(initializerType);
    if (it == registry.end())
        throw runtime_error("Unknown initializer type: " + initializerType);

    Deserializer deserializer = it->second;
    return deserializer(j);
}

}  // namespace Thor
