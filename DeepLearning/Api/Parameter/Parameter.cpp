#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <stdexcept>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {
class ApiBackedImplementationParameter : public ThorImplementation::Parameter {
   public:
    ApiBackedImplementationParameter(std::string name,
                                     bool trainable,
                                     Thor::Parameter::StorageContextStorageFactory storageContextCreateStorage)
        : ThorImplementation::Parameter(std::move(name), trainable), storageContextCreateStorage(std::move(storageContextCreateStorage)) {}

    void createStorage(const ThorImplementation::Parameter::StorageContext &context) override {
        // Either use the bound function (C++ or python) that gets input tensor context,
        // or when it was not supplied then assert that dtype and shape were supplied and create
        // a tensor co-located with the input tensors having dtype and shape.
        if (storageContextCreateStorage) {
            storage = storageContextCreateStorage(context);
        } else {
            if (dtype.isEmpty())
                throw runtime_error("Parameter.dtype was not set for parameter " + name +
                                    " and create_storage_from_context(StorageContext context) was not bound. You need to pick one of those "
                                    "routes to allocate parameter storage.");
            if (shape.isEmpty())
                throw runtime_error("Parameter.shape was not set for parameter " + name +
                                    " and create_storage_from_context(StorageContext context) was not bound. You need to pick one of those "
                                    "routes to allocate parameter storage.");
            ThorImplementation::Parameter::createStorage(context);
        }
    }

   private:
    Thor::Parameter::StorageContextStorageFactory storageContextCreateStorage;
};
}  // namespace

Parameter::Parameter(std::string name,
                     const std::vector<uint64_t> &shape,
                     DataType dtype,
                     std::shared_ptr<Initializer> initializer,
                     bool trainable,
                     std::shared_ptr<Optimizer> optimizer)
    : initialized(true),
      name(std::move(name)),
      initializer(std::move(initializer)),
      trainable(trainable),
      optimizer(std::move(optimizer)),
      trainingEnabled(trainable),
      shape(shape),
      dtype(dtype) {
    validateReadyForUse();
}

Parameter::Parameter(std::string name,
                     StorageContextStorageFactory createStorage,
                     std::shared_ptr<Initializer> initializer,
                     bool trainable,
                     std::shared_ptr<Optimizer> optimizer)
    : initialized(true),
      name(std::move(name)),
      initializer(std::move(initializer)),
      trainable(trainable),
      optimizer(std::move(optimizer)),
      trainingEnabled(trainable),
      storageContextCreateStorage(std::move(createStorage)) {
    validateReadyForUse();
}

void Parameter::validateShape(const std::vector<uint64_t> &shape) {
    if (shape.empty())
        throw runtime_error("Parameter shape cannot be empty.");
    for (uint64_t dim : shape) {
        if (dim == 0)
            throw runtime_error("Parameter shape dimensions must be > 0.");
    }
}

void Parameter::validateReadyForUse() const {
    if (name.empty())
        throw runtime_error("Parameter name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_')
        throw runtime_error("Parameter names cannot start with __; that prefix is reserved. Parameter name " + name + " is illegal.");
    if (optimizer != nullptr && !trainable)
        throw runtime_error("Only trainable parameters may have optimizer overrides.");
}

void Parameter::validateStorageFactoryReadyForStamping() const {
    const bool hasStorageContextFactory = static_cast<bool>(storageContextCreateStorage);

    if (!hasStorageContextFactory) {
        throw runtime_error("A StorageContext parameter storage factory must be bound before stamping the parameter '" + name + "'.");
    }
}

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

    // Below pattern wrong. Get the layer and assert that it casts to a parameterizable.
    // Actually the whole pattern is wrong. I need to serialize each parameter when serializing a layer,
    // like serializing a tensor. Then can just check in general if a layer supports parameters through casting
    // as a parameterizable - if even needed, because I suppose I know ahead of time which layers are parameterizable.
    // PARAMETERIZABLE DOES NOT OWN SERIALIZE, THE LAYER SIDE DOES.
    // shared_ptr<ThorImplementation::Parameterizable> physicalParameterizable =
    //     stampedNetwork.getPhysicalParameterizableFromApiParameterizable(owner->getId());
    // shared_ptr<ThorImplementation::Parameter> physicalParameter = physicalParameterizable->getParameter(name);
    // ThorImplementation::Tensor physicalStorage = physicalParameter->getStorage();
    //
    // string storageFile = "FIXME filename";
    // archiveWriter.addArchiveFile(storageFile, physicalStorage);
    //
    // if (hasOptimizer()) {
    //     j["optimizer"] = optimizer->serialize(archiveWriter,
    //                                           stream,
    //                                           physicalParameter->getOptimizer(),
    //                                           "paramaterizable" + to_string(owner->getId()) + "_" + name,
    //                                           saveOptimizerState);
    // }

    return j;
}

Parameter Parameter::deserialize(const json &j, std::shared_ptr<thor_file::TarReader> &archiveReader) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in Parameter::deserialize: " + j["version"].get<std::string>());

    Parameter deserialized;
    deserialized.name = j.at("name").get<std::string>();
    deserialized.storage = Tensor::deserialize(j["storage"]);
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

const std::string &Parameter::getName() const { return name; }
std::shared_ptr<Initializer> Parameter::getInitializer() const { return initializer; }

bool Parameter::isTrainable() const { return trainable; }
bool Parameter::isTrainingEnabled() const { return isTrainable() && trainingEnabled; }
void Parameter::setTrainingEnabled(bool enabled) {
    assert(isTrainable());

    throw runtime_error("Toggling parameter trainabilty on/off is not yet supported.");
    trainingEnabled = enabled;
}

bool Parameter::hasOptimizer() const { return optimizer != nullptr; }
std::shared_ptr<Optimizer> Parameter::getOptimizer() { return optimizer; }

ThorImplementation::Tensor Parameter::allocateStorage(const ThorImplementation::Tensor &inputTensor,
                                                      const std::vector<uint64_t> &shape,
                                                      DataType dtype) {
    validateShape(shape);
    return ThorImplementation::Parameter::allocateStorage(inputTensor.getPlacement(), shape, dtype);
}

std::shared_ptr<ThorImplementation::Parameter> Parameter::stamp() {
    validateStorageFactoryReadyForStamping();

    std::shared_ptr<ApiBackedImplementationParameter> physicalParameter =
        std::make_shared<ApiBackedImplementationParameter>(name, trainable, storageContextCreateStorage);
    if (initializer != nullptr)
        physicalParameter->setInitializer(initializer->stamp());
    return physicalParameter;
}

std::shared_ptr<Parameter> Parameter::Builder::build() {
    assert(!_name.empty());
    assert(_initializer != nullptr);
    assert(_trainable.isPresent());
    if (_optimizerOverride != nullptr)
        assert(_trainable.get() == true);

    if (!_storageContextCreateStorage) {
        throw runtime_error("Parameter::Builder requires createStorage(StorageContext) to be bound.");
    }

    return std::make_shared<Parameter>(_name, _storageContextCreateStorage, _initializer->clone(), _trainable.get(), _optimizerOverride);
}

Parameter::Builder &Parameter::Builder::name(const std::string &_name) {
    assert(this->_name.empty());
    assert(!_name.empty());
    if (_name.length() >= 2 && _name[0] == '_' && _name[1] == '_')
        throw runtime_error("Parameter names cannot start with __ that is reserved. Parameter name " + _name + " is illegal.");
    this->_name = _name;
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

Parameter::Builder &Parameter::Builder::createStorage(StorageContextStorageFactory createStorage) {
    if (this->_storageContextCreateStorage) {
        throw runtime_error("Parameter::Builder storage factory may only be bound once.");
    }
    this->_storageContextCreateStorage = std::move(createStorage);
    return *this;
}

}  // namespace Thor
