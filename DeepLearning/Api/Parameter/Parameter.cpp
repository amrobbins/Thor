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
    explicit ApiBackedImplementationParameter(const Thor::Parameter *apiParameter)
        : ThorImplementation::Parameter(apiParameter->getName(), apiParameter->isTrainable()), apiParameter(apiParameter) {
        if (this->apiParameter == nullptr)
            throw runtime_error("Cannot stamp a null API Parameter.");
    }

    void createStorage(const ThorImplementation::Parameter::StorageContext &context) override {
        storage = apiParameter->create_storage(context);
    }
    void createStorage(const ThorImplementation::Tensor &inputTensor) override { storage = apiParameter->create_storage(inputTensor); }

   private:
    const Thor::Parameter *apiParameter;
};
}  // namespace

Parameter::Parameter(std::string name,
                     std::vector<uint64_t> shape,
                     DataType dtype,
                     std::shared_ptr<Initializer> initializer,
                     bool trainable,
                     std::shared_ptr<Optimizer> optimizer)
    : initialized(true),
      name(std::move(name)),
      shape(std::move(shape)),
      dtype(dtype),
      initializer(std::move(initializer)),
      trainable(trainable),
      optimizer(std::move(optimizer)),
      trainingEnabled(trainable) {
    validateReadyForUse();
    storage = Tensor(this->dtype, this->shape);
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
    validateShape(shape);
    if (optimizer != nullptr && !trainable)
        throw runtime_error("Only trainable parameters may have optimizer overrides.");
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

const std::string &Parameter::getName() const { return name; }
const std::vector<uint64_t> &Parameter::getShape() const { return shape; }
Parameter::DataType Parameter::getDataType() const { return dtype; }
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

ThorImplementation::Tensor Parameter::createStorage(const StorageContext &context) const {
    return createStorage(context.getFeatureInput());
}

ThorImplementation::Tensor Parameter::createStorage(const ThorImplementation::Tensor &inputTensor) const {
    if (!initialized)
        throw runtime_error("Cannot create storage for an uninitialized Parameter.");
    validateReadyForUse();

    return createStorage(inputTensor, shape, dtype);
}

ThorImplementation::Tensor Parameter::createStorage(const StorageContext &context,
                                                    const std::vector<uint64_t> &shape,
                                                    DataType dtype) const {
    return createStorage(context.getFeatureInput(), shape, dtype);
}

ThorImplementation::Tensor Parameter::createStorage(const ThorImplementation::Tensor &inputTensor,
                                                    const std::vector<uint64_t> &shape,
                                                    DataType dtype) const {
    validateShape(shape);
    ThorImplementation::TensorDescriptor descriptor(dtype, shape);
    return ThorImplementation::Tensor(inputTensor.getPlacement(), descriptor);
}

ThorImplementation::Tensor Parameter::create_storage(const StorageContext &context) const { return createStorage(context); }

ThorImplementation::Tensor Parameter::create_storage(const ThorImplementation::Tensor &inputTensor) const {
    return createStorage(inputTensor);
}

std::shared_ptr<ThorImplementation::Parameter> Parameter::stamp() {
    std::shared_ptr<ApiBackedImplementationParameter> physicalParameter = std::make_shared<ApiBackedImplementationParameter>(this);
    if (initializer != nullptr)
        physicalParameter->setInitializer(initializer->stamp());
    return physicalParameter;
}

std::shared_ptr<Parameter> Parameter::Builder::build() {
    assert(!_name.empty());
    assert(!_shape.empty());
    if (_dtype.isEmpty())
        _dtype = DataType::FP32;
    assert(_initializer != nullptr);
    assert(_trainable.isPresent());
    if (_optimizerOverride != nullptr)
        assert(_trainable.get() == true);

    std::shared_ptr<Parameter> parameter =
        std::make_shared<Parameter>(_name, _shape, _dtype.get(), _initializer->clone(), _trainable.get(), _optimizerOverride);
    return parameter;
}

Parameter::Builder &Parameter::Builder::name(const std::string &_name) {
    assert(this->_name.empty());
    assert(!_name.empty());
    if (_name.length() >= 2 && _name[0] == '_' && _name[1] == '_')
        throw runtime_error("Parameter names cannot start with __ that is reserved. Parameter name " + _name + " is illegal.");
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

}  // namespace Thor
