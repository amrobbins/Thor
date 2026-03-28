#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {

json Parameter::architectureJson() const {
    json j;
    j["version"] = getVersion();
    j["name"] = name;
    j["storage"] = storage.architectureJson();
    j["trainable"] = trainable;
    if (trainable)
        j["training_enabled"] = trainingEnabled;
    if (initializer != nullptr)
        j["initializer"] = initializer->architectureJson();
    if (optimizer != nullptr)
        j["optimizer_override"] = optimizer->architectureJson();
    return j;
}

json Parameter::serialize(thor_file::TarWriter &archiveWriter,
                          Stream stream,
                          bool saveOptimizerState,
                          ThorImplementation::StampedNetwork &stampedNetwork) const {
    json j = architectureJson();

    shared_ptr<ThorImplementation::Parameterizable> physicalParameterizable =
        stampedNetwork.getPhysicalParameterizableFromApiParameterizable(owner->getId());
    shared_ptr<ThorImplementation::Parameter> physicalParameter = physicalParameterizable->getParam(name);
    ThorImplementation::Tensor physicalStorage = physicalParameter->getStorage();

    string storageFile = "FIXME filename";
    archiveWriter.addArchiveFile(storageFile, physicalStorage);

    if (hasOptimizer()) {
        j["optimizer"] = optimizer->serialize(archiveWriter,
                                              stream,
                                              physicalParameter->getOptimizer(),
                                              "paramaterizable" + to_string(owner->getId()) + "_" + name,
                                              saveOptimizerState);
    }

    return j;
}

Parameter Parameter::deserialize(const json &j, std::shared_ptr<thor_file::TarReader> &archiveReader) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in Parameter::deserialize: " + j["version"].get<std::string>());

    Parameter deserialized;
    deserialized.name = j.at("name").get<std::string>();
    deserialized.storage = Tensor::deserialize(j["storage"]);
    deserialized.shape = deserialized.storage.getDimensions();
    deserialized.dtype = deserialized.storage.getDataType();
    deserialized.trainable = j.at("trainable").get<bool>();
    if (deserialized.trainable)
        deserialized.trainingEnabled = j.at("training_enabled").get<bool>();
    else
        deserialized.trainingEnabled = false;
    if (j.contains("initializer"))
        deserialized.initializer = Initializer::deserialize(j["initializer"]);
    if (j.contains("optimizer_override"))
        deserialized.optimizer = Optimizer::deserialize(archiveReader, j["optimizer_override"], nullptr);
    deserialized.initialized = true;
    return deserialized;
}
std::string Parameter::getVersion() { return "1.0.0"; }

bool Parameter::isTrainable() const { return trainable; }
bool Parameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled; }
void Parameter::setTrainingEnabled(bool enabled) {
    assert(isTrainable());
    trainingEnabled = enabled;
}

bool Parameter::hasOptimizer() const { return optimizer != nullptr; }
std::shared_ptr<Optimizer> Parameter::getOptimizer() { return optimizer; }

std::shared_ptr<Parameter> Parameter::Builder::build() {
    assert(!_name.empty());
    assert(!_shape.empty());
    if (_dtype.isEmpty())
        _dtype = DataType::FP32;
    assert(_initializer != nullptr);
    assert(_trainable.isPresent());
    if (_optimizerOverride != nullptr)
        assert(_trainable.get() == true);

    std::shared_ptr<Parameter> parameter = std::make_shared<Parameter>();
    parameter->name = _name;
    parameter->shape = _shape;
    parameter->dtype = _dtype;
    parameter->initializer = _initializer;
    parameter->trainable = _trainable;
    parameter->trainingEnabled = parameter->trainable;
    parameter->optimizer = _optimizerOverride;
    parameter->storage = Tensor(parameter->dtype, parameter->shape);
    parameter->initialized = true;
    parameter->owner = _owner;
    return parameter;
}

Parameter::Builder &Parameter::Builder::name(const std::string &_name) {
    assert(this->_name.empty());
    assert(!_name.empty());
    this->_name = _name;
    return *this;
}

Parameter::Builder &Parameter::Builder::shape(const std::vector<uint64_t> &_shape) {
    assert(this->_shape.empty());
    assert(!_shape.empty());
    this->_shape = _shape;
    return *this;
}

Parameter::Builder &Parameter::Builder::dtype(const DataType &_dtype) {
    assert(!this->_dtype.isPresent());
    this->_dtype = _dtype;
    return *this;
}

Parameter::Builder &Parameter::Builder::initializer(std::shared_ptr<Initializer> &_initializer) {
    assert(this->_initializer == nullptr);
    this->_initializer = _initializer->clone();
    return *this;
}

Parameter::Builder &Parameter::Builder::initializer(std::shared_ptr<Initializer> &&_initializer) {
    assert(this->_initializer == nullptr);
    this->_initializer = _initializer->clone();
    return *this;
}

Parameter::Builder &Parameter::Builder::trainable(const bool _trainable) {
    assert(!this->_trainable.isPresent());
    this->_trainable = _trainable;
    return *this;
}

Parameter::Builder &Parameter::Builder::optimizer(std::shared_ptr<Optimizer> &_optimizerOverride) {
    assert(this->_optimizerOverride == nullptr);
    this->_optimizerOverride = _optimizerOverride;
    return *this;
}

Parameter::Builder &Parameter::Builder::optimizer(std::shared_ptr<Optimizer> &&_optimizerOverride) {
    assert(this->_optimizerOverride == nullptr);
    this->_optimizerOverride = _optimizerOverride;
    return *this;
};

Parameter::Builder &Parameter::Builder::owner(const std::shared_ptr<Parameterizable> &_owner) {
    assert(this->_owner == nullptr);
    this->_owner = _owner;
    return *this;
};

}  // namespace Thor
