#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"

#include "DeepLearning/Implementation/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include <optional>
#include "Utilities/TarFile/TarReader.h"

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
        // or when it was not supplied then verify that dtype and shape were supplied and create
        // a tensor co-located with the input tensors having dtype and shape.
        if (storageContextCreateStorage) {
            storage = storageContextCreateStorage(context);
        } else {
            if (!dtype.has_value())
                throw runtime_error("Parameter.dtype was not set for parameter " + name +
                                    " and create_storage_from_context(StorageContext context) was not bound. You need to pick one of those "
                                    "routes to allocate parameter storage.");
            if (!shape.has_value())
                throw runtime_error("Parameter.shape was not set for parameter " + name +
                                    " and create_storage_from_context(StorageContext context) was not bound. You need to pick one of those "
                                    "routes to allocate parameter storage.");
            ThorImplementation::PhysicalParameter::createStorage(context);
        }
    }

   private:
    Thor::ParameterSpecification::StorageContextStorageFactory storageContextCreateStorage;
};

class ArchiveTensorInitializer : public ThorImplementation::Initializer {
   public:
    ArchiveTensorInitializer(std::shared_ptr<thor_file::TarReader> archiveReader, std::string fileName)
        : archiveReader(std::move(archiveReader)), fileName(std::move(fileName)) {
        if (this->archiveReader == nullptr) {
            throw std::runtime_error("ArchiveTensorInitializer requires a TarReader.");
        }
        if (this->fileName.empty()) {
            throw std::runtime_error("ArchiveTensorInitializer requires a non-empty file name.");
        }
    }

    void initialize(Stream initStream) override {
        (void)initStream;
        if (archiveReader == nullptr) {
            throw std::runtime_error("ArchiveTensorInitializer cannot initialize after its archive reader has been consumed.");
        }
        archiveReader->registerReadRequest(fileName, weights);
        archiveReader = nullptr;
        fileName.clear();
    }

    std::shared_ptr<ThorImplementation::Initializer> clone() override { return std::make_shared<ArchiveTensorInitializer>(*this); }

   private:
    std::shared_ptr<thor_file::TarReader> archiveReader;
    std::string fileName;
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
    if (initializer == nullptr)
        throw runtime_error("All parameters require an initializer, initializer is nullptr.");
    if (optimizer != nullptr && !trainable)
        throw runtime_error("Only trainable parameters may have optimizer overrides.");
}

void ParameterSpecification::validateStorageFactoryReadyForStamping() const {
    if (dtype.has_value() && shape.has_value())
        return;

    const bool hasStorageContextFactory = static_cast<bool>(storageContextCreateStorage);
    if (!hasStorageContextFactory) {
        throw runtime_error("A StorageContext parameter storage factory must be bound before stamping the parameter '" + name + "'.");
    }
}

// Parameters don't need to be serialized, bound parameters do. That will resolve the trainingInitiallyEnabled vs current state issue.
json ParameterSpecification::architectureJson() const {
    // Here I call architectureJson for parameter, initializer and optimizer

    json j;
    j["version"] = getVersion();
    j["name"] = name;
    if (storage.isInitialized()) {
        j["storage"] = storage.architectureJson();
    } else if (shape.has_value() && dtype.has_value()) {
        j["shape"] = shape.value();
        j["dtype"] = json(dtype.value());
    } else {
        // shape and dtype should always be present by the time serialize is called,
        // because the layer has been connected to the network, and that is when this info is determined,
        // before the network has been serialized.
        // The following indicates an architecture bug:
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

ParameterSpecification ParameterSpecification::deserialize(const json &j, std::shared_ptr<thor_file::TarReader> &archiveReader) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw std::runtime_error("Unsupported version in ParameterSpecification::deserialize: " + j["version"].get<std::string>());

    ParameterSpecification deserialized;
    deserialized.name = j.at("name").get<std::string>();
    if (j.contains("storage")) {
        deserialized.storage = Tensor::deserialize(j["storage"], archiveReader.get());
        deserialized.shape = deserialized.storage.getDimensions();
        deserialized.dtype = deserialized.storage.getDataType();
    } else {
        if (!j.contains("shape") || !j.contains("dtype")) {
            throw std::runtime_error(
                "ParameterSpecification::deserialize requires either a serialized storage tensor or both shape and dtype metadata.");
        }
        deserialized.shape = j.at("shape").get<std::vector<uint64_t>>();
        deserialized.dtype = j.at("dtype").get<DataType>();
    }

    const char *storageFileKey = nullptr;
    if (j.contains("storage_file")) {
        storageFileKey = "storage_file";
    } else if (j.contains("parameter_storage")) {
        // Backward-compatible while the old key is phased out.
        storageFileKey = "parameter_storage";
    }
    if (storageFileKey != nullptr) {
        deserialized.archiveReader = archiveReader;
        deserialized.storageFile = j.at(storageFileKey).get<std::string>();
    }

    deserialized.trainable = j.at("trainable").get<bool>();
    if (deserialized.trainable)
        deserialized.trainingInitiallyEnabled = j.value("training_enabled", true);
    else
        deserialized.trainingInitiallyEnabled = false;
    if (j.contains("initializer"))
        deserialized.initializer = Initializer::deserialize(j["initializer"]);
    if (j.contains("optimizer_override"))
        deserialized.optimizer = Optimizer::deserialize(archiveReader, j["optimizer_override"], nullptr);
    else if (j.contains("optimizer"))
        deserialized.optimizer = Optimizer::deserialize(archiveReader, j["optimizer"], nullptr);
    deserialized.initialized = true;
    deserialized.validateReadyForUse();
    return deserialized;
}
std::string ParameterSpecification::getVersion() { return "1.0.0"; }

const std::string &ParameterSpecification::getName() const { return name; }
std::shared_ptr<Initializer> ParameterSpecification::getInitializer() const { return initializer; }

bool ParameterSpecification::isTrainable() const { return trainable; }
bool ParameterSpecification::isTrainingInitiallyEnabled() const { return isTrainable() && trainingInitiallyEnabled; }

void ParameterSpecification::setTrainingInitiallyEnabled(bool enabled) {
    if (!isTrainable()) {
        throw runtime_error("Only trainable parameters may toggle training enabled. Parameter '" + name + "' is not trainable.");
    }
    trainingInitiallyEnabled = enabled;
}

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
        THOR_THROW_IF_FALSE(shape.has_value());
        THOR_THROW_IF_FALSE(dtype.has_value());

        physicalParameter = std::make_shared<ThorImplementation::PhysicalParameter>(name, trainable, shape.value(), dtype.value());
    }

    if (storageFile.has_value()) {
        std::shared_ptr<ThorImplementation::Initializer> archiveInitializer =
            std::make_shared<ArchiveTensorInitializer>(archiveReader, storageFile.value());
        physicalParameter->setInitializer(archiveInitializer);
    } else if (initializer != nullptr) {
        physicalParameter->setInitializer(initializer->stamp());
    }

    if (optimizer != nullptr) {
        physicalParameter->setOptimizer(optimizer->stamp(nullptr));
    }

    if (trainable && !trainingInitiallyEnabled) {
        physicalParameter->setTrainingEnabled(false);
    }

    return physicalParameter;
}

bool ParameterSpecification::setOptimizer(const std::shared_ptr<Optimizer> &optimizer, bool override) {
    if (override || this->optimizer == nullptr) {
        this->optimizer = optimizer->clone();
        return true;
    }
    return false;
}

ParameterSpecification ParameterSpecification::Builder::build() {
    THOR_THROW_IF_FALSE(!_name.empty());
    THOR_THROW_IF_FALSE(_trainable.has_value());
    THOR_THROW_IF_FALSE(_initializer != nullptr);

    ParameterSpecification p;
    p.initialized = true;
    p.name = _name;
    p.trainable = _trainable.value();
    p.initializer = _initializer->clone();
    if (p.trainable) {
        if (!_trainingInitiallyEnabled.has_value() || _trainingInitiallyEnabled.value() == true)
            p.trainingInitiallyEnabled = true;
        else
            p.trainingInitiallyEnabled = false;
    } else {
        if (_trainingInitiallyEnabled.has_value() && _trainingInitiallyEnabled.value() == true)
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
        if (!_dtype.has_value()) {
            throw runtime_error(
                "ParameterSpecification::Builder when createStorage(StorageContextStorageFactory) is not bound, then a dtype is required, "
                "but no dtype was specified.");
        }
        if (_shape.empty()) {
            throw runtime_error(
                "ParameterSpecification::Builder when createStorage(StorageContextStorageFactory) is not bound, then a shape is required, "
                "but no shape was specified.");
        }
        p.dtype = _dtype.value();
        p.shape = _shape;
    } else {
        p.storageContextCreateStorage = _storageContextCreateStorage;
    }

    p.validateReadyForUse();
    return p;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::name(const std::string &_name) {
    THOR_THROW_IF_FALSE(this->_name.empty());
    THOR_THROW_IF_FALSE(!_name.empty());
    if (_name.length() >= 2 && _name[0] == '_' && _name[1] == '_')
        throw runtime_error("Parameter names cannot start with __ that is reserved. Parameter name " + _name + " is illegal.");
    this->_name = _name;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::initializer(std::shared_ptr<Initializer> &_initializer) {
    THOR_THROW_IF_FALSE(this->_initializer == nullptr);
    this->_initializer = _initializer->clone();
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::initializer(std::shared_ptr<Initializer> &&_initializer) {
    THOR_THROW_IF_FALSE(this->_initializer == nullptr);
    this->_initializer = _initializer->clone();
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::trainable(const bool _trainable) {
    THOR_THROW_IF_FALSE(!this->_trainable.has_value());
    this->_trainable = _trainable;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::trainingInitiallyEnabled(const bool enabled) {
    if (this->_trainingInitiallyEnabled.has_value()) {
        throw runtime_error("ParameterSpecification::Builder trainingInitiallyEnabled may only be specified once.");
    }
    this->_trainingInitiallyEnabled = enabled;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::optimizer(std::shared_ptr<Optimizer> &_optimizerOverride) {
    THOR_THROW_IF_FALSE(this->_optimizerOverride == nullptr);
    this->_optimizerOverride = _optimizerOverride;
    return *this;
}

ParameterSpecification::Builder &ParameterSpecification::Builder::optimizer(std::shared_ptr<Optimizer> &&_optimizerOverride) {
    THOR_THROW_IF_FALSE(this->_optimizerOverride == nullptr);
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
    if (this->_dtype.has_value()) {
        throw runtime_error("ParameterSpecification::Builder dtype may only be specified once.");
    }
    this->_dtype = _dtype;
    return *this;
}

}  // namespace Thor
