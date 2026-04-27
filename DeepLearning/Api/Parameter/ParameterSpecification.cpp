#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <stdexcept>
#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

namespace {
class ApiBackedImplementationParameter : public ThorImplementation::PhysicalParameter {
   public:
    ApiBackedImplementationParameter(std::string name,
                                     bool trainable,
                                     Thor::ParameterSpecification::StorageContextStorageFactory storageContextCreateStorage)
        : ThorImplementation::PhysicalParameter(std::move(name), trainable),
          storageContextCreateStorage(std::move(storageContextCreateStorage)) {}

    void createStorage(const ThorImplementation::PhysicalParameter::StorageContext &context) override {
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
            ThorImplementation::PhysicalParameter::createStorage(context);
        }
    }

   private:
    Thor::ParameterSpecification::StorageContextStorageFactory storageContextCreateStorage;
};
}  // namespace

ParameterSpecification::ParameterSpecification(std::string name,
                                               const std::vector<uint64_t> &shape,
                                               DataType dtype,
                                               std::shared_ptr<Initializer> initializer,
                                               bool trainable,
                                               std::shared_ptr<Optimizer> optimizer,
                                               bool trainingInitiallyEnabled)
    : initialized(true),
      name(std::move(name)),
      initializer(std::move(initializer)),
      trainable(trainable),
      optimizer(std::move(optimizer)),
      trainingInitiallyEnabled(trainingInitiallyEnabled),
      shape(shape),
      dtype(dtype) {
    validateReadyForUse();
}

ParameterSpecification::ParameterSpecification(std::string name,
                                               StorageContextStorageFactory createStorage,
                                               std::shared_ptr<Initializer> initializer,
                                               bool trainable,
                                               std::shared_ptr<Optimizer> optimizer,
                                               bool trainingInitiallyEnabled)
    : initialized(true),
      name(std::move(name)),
      initializer(std::move(initializer)),
      trainable(trainable),
      optimizer(std::move(optimizer)),
      trainingInitiallyEnabled(trainingInitiallyEnabled),
      storageContextCreateStorage(std::move(createStorage)) {
    validateReadyForUse();
}

void ParameterSpecification::validateShape(const std::vector<uint64_t> &shape) {
    if (shape.empty())
        throw runtime_error("Parameter shape cannot be empty.");
    for (uint64_t dim : shape) {
        if (dim == 0)
            throw runtime_error("Parameter shape dimensions must be > 0.");
    }
}

void ParameterSpecification::validateReadyForUse() const {
    if (name.empty())
        throw runtime_error("Parameter name cannot be empty.");
    if (name.length() >= 2 && name[0] == '_' && name[1] == '_')
        throw runtime_error("Parameter names cannot start with __; that prefix is reserved. Parameter name " + name + " is illegal.");
    if (optimizer != nullptr && !trainable)
        throw runtime_error("Only trainable parameters may have optimizer overrides.");
}

void ParameterSpecification::validateStorageFactoryReadyForStamping() const {
    if (dtype.isPresent() && shape.isPresent())
        return;

    const bool hasStorageContextFactory = static_cast<bool>(storageContextCreateStorage);
    if (!hasStorageContextFactory) {
        throw runtime_error("A StorageContext parameter storage factory must be bound before stamping the parameter '" + name + "'.");
    }
}

// Parameters don't need to be serialized, bound parameters do. That will resolve the trainingInitiallyEnabled vs current state issue.
json ParameterSpecification::architectureJson() const {
    json j;
    j["version"] = getVersion();
    j["name"] = name;
    j["storage"] = storage.architectureJson();
    j["trainable"] = trainable;
    if (trainable)
        j["training_enabled"] = trainingInitiallyEnabled;
    if (initializer != nullptr)
        j["initializer"] = initializer->architectureJson();
    if (optimizer != nullptr)
        j["optimizer_override"] = optimizer->architectureJson();
    return j;
}

json ParameterSpecification::serialize(thor_file::TarWriter &archiveWriter,
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

ParameterSpecification ParameterSpecification::deserialize(const json &j, std::shared_ptr<thor_file::TarReader> &archiveReader) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in Parameter::deserialize: " + j["version"].get<std::string>());

    ParameterSpecification deserialized;
    deserialized.name = j.at("name").get<std::string>();
    deserialized.storage = Tensor::deserialize(j["storage"]);
    deserialized.trainable = j.at("trainable").get<bool>();
    if (deserialized.trainable)
        deserialized.trainingInitiallyEnabled = j.at("training_enabled").get<bool>();
    else
        deserialized.trainingInitiallyEnabled = false;
    if (j.contains("initializer"))
        deserialized.initializer = Initializer::deserialize(j["initializer"]);
    if (j.contains("optimizer_override"))
        deserialized.optimizer = Optimizer::deserialize(archiveReader, j["optimizer_override"], nullptr);
    deserialized.initialized = true;
    return deserialized;
}
std::string ParameterSpecification::getVersion() { return "1.0.0"; }

const std::string &ParameterSpecification::getName() const { return name; }
std::shared_ptr<Initializer> ParameterSpecification::getInitializer() const { return initializer; }

bool ParameterSpecification::isTrainable() const { return trainable; }
bool ParameterSpecification::isTrainingInitiallyEnabled() const { return isTrainable() && trainingInitiallyEnabled; }

bool ParameterSpecification::hasOptimizer() const { return optimizer != nullptr; }
std::shared_ptr<Optimizer> ParameterSpecification::getOptimizer() { return optimizer; }

ThorImplementation::Tensor ParameterSpecification::allocateStorage(const ThorImplementation::Tensor &inputTensor,
                                                                   const std::vector<uint64_t> &shape,
                                                                   DataType dtype) {
    validateShape(shape);
    return ThorImplementation::PhysicalParameter::allocateStorage(inputTensor.getPlacement(), shape, dtype);
}

std::shared_ptr<ThorImplementation::PhysicalParameter> ParameterSpecification::stamp() {
    validateStorageFactoryReadyForStamping();

    std::shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter;

    if (storageContextCreateStorage) {
        physicalParameter = std::make_shared<ApiBackedImplementationParameter>(name, trainable, storageContextCreateStorage);
    } else {
        assert(shape.isPresent());
        assert(dtype.isPresent());

        physicalParameter = std::make_shared<ThorImplementation::PhysicalParameter>(name, trainable, shape.get(), dtype.get());
    }

    if (initializer != nullptr) {
        physicalParameter->setInitializer(initializer->stamp());
    }

    if (trainable && !trainingInitiallyEnabled) {
        physicalParameter->setTrainingEnabled(false);
    }

    return physicalParameter;
}

std::shared_ptr<ParameterSpecification> ParameterSpecification::Builder::build() {
    assert(!_name.empty());
    assert(_initializer != nullptr);
    assert(_trainable.isPresent());
    if (_optimizerOverride != nullptr)
        assert(_trainable.get() == true);

    if (!_storageContextCreateStorage) {
        throw runtime_error("Parameter::Builder requires createStorage(StorageContext) to be bound.");
    }

    return std::make_shared<ParameterSpecification>(
        _name, _storageContextCreateStorage, _initializer->clone(), _trainable.get(), _optimizerOverride);
}

ParameterSpecification::Builder &ParameterSpecification::Builder::name(const std::string &_name) {
    assert(this->_name.empty());
    assert(!_name.empty());
    if (_name.length() >= 2 && _name[0] == '_' && _name[1] == '_')
        throw runtime_error("Parameter names cannot start with __ that is reserved. Parameter name " + _name + " is illegal.");
    this->_name = _name;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::initializer(std::shared_ptr<Initializer> &_initializer) {
    assert(this->_initializer == nullptr);
    this->_initializer = _initializer->clone();
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::initializer(std::shared_ptr<Initializer> &&_initializer) {
    assert(this->_initializer == nullptr);
    this->_initializer = _initializer->clone();
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::trainable(const bool _trainable) {
    assert(!this->_trainable.isPresent());
    this->_trainable = _trainable;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::optimizer(std::shared_ptr<Optimizer> &_optimizerOverride) {
    assert(this->_optimizerOverride == nullptr);
    this->_optimizerOverride = _optimizerOverride;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::optimizer(std::shared_ptr<Optimizer> &&_optimizerOverride) {
    assert(this->_optimizerOverride == nullptr);
    this->_optimizerOverride = _optimizerOverride;
    return *this;
};

ParameterSpecification::Builder &ParameterSpecification::Builder::createStorage(StorageContextStorageFactory createStorage) {
    if (this->_storageContextCreateStorage) {
        throw runtime_error("Parameter::Builder storage factory may only be bound once.");
    }
    this->_storageContextCreateStorage = std::move(createStorage);
    return *this;
}

}  // namespace Thor
