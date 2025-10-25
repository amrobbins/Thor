#include "DeepLearning/Api/Tensor/Tensor.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

atomic<uint64_t> Tensor::nextId(10000);

json Tensor::serialize() const {
    return json{{"version", "1.0.0"}, {"id", getId()}, {"dimensions", getDimensions()}, {"data_type", json(getDataType())}};
}

Tensor Tensor::deserialize(const json &j) {
    uint32_t originalId = j.at("id").get<uint64_t>();
    vector<uint64_t> dimensions = j.at("dimensions").get<vector<uint64_t>>();
    DataType dataType = j.at("data_type").get<DataType>();
    Tensor deserialized(dataType, dimensions);
    deserialized.originalId = originalId;
    deserialized.initialized = true;
    return deserialized;
}
