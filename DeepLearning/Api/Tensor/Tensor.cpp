#include "DeepLearning/Api/Tensor/Tensor.h"

#include <stdexcept>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

atomic<uint64_t> Tensor::nextId(10000);

json Tensor::architectureJson() const {
    return json{{"version", getVersion()}, {"id", getId()}, {"dimensions", getDimensions()}, {"data_type", json(getDataType())}};
}

json Tensor::serialize(thor_file::TarWriter &archiveWriter) const {
    // API tensors currently have no binary payload of their own. Keeping serialize() here makes tensor
    // ownership explicit and lets future tensor-attached state be added without changing layer code.
    (void)archiveWriter;
    return architectureJson();
}

Tensor Tensor::deserialize(const json &j, thor_file::TarReader *archiveReader) {
    (void)archiveReader;
    if (j.at("version").get<std::string>() != "1.0.0") {
        throw std::runtime_error("Unsupported version in Tensor::deserialize: " + j.at("version").get<std::string>());
    }
    uint64_t originalId = j.at("id").get<uint64_t>();
    vector<uint64_t> dimensions = j.at("dimensions").get<vector<uint64_t>>();
    DataType dataType = j.at("data_type").get<DataType>();
    Tensor deserialized(dataType, dimensions);
    deserialized.originalId = originalId;
    deserialized.initialized = true;
    return deserialized;
}
