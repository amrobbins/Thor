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
    if (storage.isInitialized()) {
        j["storage"] = storage.architectureJson();
    } else if (shape.isPresent() && dtype.isPresent()) {
        j["shape"] = shape.get();
        j["dtype"] = json(dtype.get());
    } else {
        throw runtime_error(
            "Parameter '" + name +
            "' is not serializable because its storage is determined by a runtime StorageContext factory and no resolved storage "
            "or static shape/dtype definition is available.");
    }
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
    // shared_ptr<ThorImplementation::PhysicalParameterizable> physicalParameterizable =
    //     stampedNetwork.getPhysicalParameterizableFromApiParameterizable(owner->getId());
    // shared_ptr<ThorImplementation::PhysicalParameter> physicalParameter = physicalParameterizable->getParameter(name);
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
        throw std::runtime_error("Unsupported version in ParameterSpecification::deserialize: " + j["version"].get<std::string>());

    ParameterSpecification deserialized;
    deserialized.name = j.at("name").get<std::string>();
    if (j.contains("storage")) {
        deserialized.storage = Tensor::deserialize(j["storage"]);
    } else {
        if (!j.contains("shape") || !j.contains("dtype")) {
            throw std::runtime_error(
                "ParameterSpecification::deserialize requires either a serialized storage tensor or both shape and dtype metadata.");
        }
        deserialized.shape = j.at("shape").get<std::vector<uint64_t>>();
        deserialized.dtype = j.at("dtype").get<DataType>();
    }
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

ParameterSpecification ParameterSpecification::Builder::build() {
    assert(!_name.empty());
    assert(_trainable.isPresent());
    assert(_initializer != nullptr);

    ParameterSpecification p;
    p.name = _name;
    p.trainable = _trainable;
    p.initializer = _initializer->clone();
    if (p.trainable) {
        if (_trainingInitiallyEnabled.isEmpty() || _trainingInitiallyEnabled.get() == true)
            p.trainingInitiallyEnabled = true;
        else
            p.trainingInitiallyEnabled = false;
    } else {
        if (_trainingInitiallyEnabled.isPresent() && _trainingInitiallyEnabled.get() == true)
            throw runtime_error("trainingInitiallyEnabled set to true for parameter named " + p.name +
                                " but the parameter has trainable == false");
        p.trainingInitiallyEnabled = false;
    }
    if (_optimizerOverride == nullptr) {
        p.optimizer = nullptr;
    } else {
        p.optimizer = _optimizerOverride.get()->clone();
    }
    if (!_storageContextCreateStorage) {
        if (_dtype.isEmpty()) {
            throw runtime_error(
                "ParameterSpecification::Builder when createStorage(StorageContextStorageFactory) is not bound, then a dtype is required, "
                "but no dtype was specified.");
        }
        if (_shape.empty()) {
            throw runtime_error(
                "ParameterSpecification::Builder when createStorage(StorageContextStorageFactory) is not bound, then a shape is required, "
                "but no shape was specified.");
        }
        p.dtype = _dtype;
        p.shape = _shape;
    } else {
        p.storageContextCreateStorage = _storageContextCreateStorage;
    }

    return p;
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

ParameterSpecification::Builder &ParameterSpecification::Builder::trainingInitiallyEnabled(const bool enabled) {
    if (this->_trainingInitiallyEnabled.isPresent()) {
        throw runtime_error("ParameterSpecification::Builder trainingInitiallyEnabled may only be specified once.");
    }
    this->_trainingInitiallyEnabled = enabled;
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

ParameterSpecification::Builder &ParameterSpecification::Builder::createStorage(StorageContextStorageFactory _storageContextCreateStorage) {
    if (this->_storageContextCreateStorage) {
        throw runtime_error("ParameterSpecification::Builder storage factory may only be bound once.");
    }
    this->_storageContextCreateStorage = std::move(_storageContextCreateStorage);
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::shape(const std::vector<uint64_t> &_shape) {
    if (_shape.empty()) {
        throw runtime_error("ParameterSpecification::Builder shape may not be empty.");
    }
    if (!this->_shape.empty()) {
        throw runtime_error("ParameterSpecification::Builder shape may only be specified once.");
    }
    for (uint32_t i = 0; i < _shape.size(); ++i) {
        if (_shape[i] == 0) {
            throw runtime_error("ParameterSpecification::Builder shape may not have dimensions of size zero. Dimension " + to_string(i) +
                                " is 0.");
        }
    }

    this->_shape = _shape;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::dtype(const DataType _dtype) {
    if (this->_dtype.isPresent()) {
        throw runtime_error("ParameterSpecification::Builder dtype may only be specified once.");
    }
    this->_dtype = _dtype;
    return *this;
}

}  // namespace Thor
