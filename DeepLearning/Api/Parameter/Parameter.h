#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Common/Optional.h"

namespace Thor {
class Optimizer;

class Parameter {
   public:
    using DataType = ThorImplementation::TensorDescriptor::DataType;
    class Builder;

    virtual ~Parameter() = default;

    virtual nlohmann::json architectureJson() const {
        nlohmann::json j;
        j["version"] = getVersion();
        j["name"] = name;
        j["storage"] = storage.architectureJson();
        j["trainable"] = trainable;
        if (trainable)
            j["training_enabled"] = trainingEnabled;
        if (initializer != nullptr)
            j["initializer"] = initializer->architectureJson();
        if (optimizerOverride != nullptr)
            j["optimizer_override"] = optimizerOverride->architectureJson();
        return j;
    }

    static Parameter deserialize(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader) {
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
            deserialized.optimizerOverride = Optimizer::deserialize(archiveReader, j["optimizer_override"], nullptr);
        deserialized.initialized = true;
        return deserialized;
    }
    static std::string getVersion() { return "1.0.0"; }

    bool isTrainable() const { return trainable; }
    bool isTrainingEnabled() const { return isTrainable() && trainingEnabled; }
    void setTrainingEnabled(bool enabled) {
        assert(isTrainable());
        trainingEnabled = enabled;
    }

   private:
    bool initialized = false;

    std::string name{};
    std::vector<uint64_t> shape{};
    DataType dtype = DataType::FP32;
    std::shared_ptr<Initializer> initializer = nullptr;
    bool trainable = false;
    std::shared_ptr<Optimizer> optimizerOverride = nullptr;
    bool trainingEnabled = false;

    Tensor storage;
};

class Parameter::Builder {
   public:
    virtual ~Builder() = default;
    virtual std::shared_ptr<Parameter> build() {
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
        parameter->optimizerOverride = _optimizerOverride;
        parameter->storage = Tensor(parameter->dtype, parameter->shape);
        parameter->initialized = true;

        return parameter;
    }

    virtual Parameter::Builder &name(const std::string &_name) {
        assert(this->_name.empty());
        assert(!_name.empty());
        this->_name = _name;
        return *this;
    }

    virtual Parameter::Builder &shape(const std::vector<uint64_t> &_shape) {
        assert(this->_shape.empty());
        assert(!_shape.empty());
        this->_shape = _shape;
        return *this;
    }

    virtual Parameter::Builder &dtype(const DataType &_dtype) {
        assert(!this->_dtype.isPresent());
        this->_dtype = _dtype;
        return *this;
    }

    virtual Parameter::Builder &initializer(std::shared_ptr<Initializer> &_initializer) {
        assert(this->_initializer == nullptr);
        this->_initializer = _initializer->clone();
        return *this;
    }

    virtual Parameter::Builder &initializer(std::shared_ptr<Initializer> &&_initializer) {
        assert(this->_initializer == nullptr);
        this->_initializer = _initializer->clone();
        return *this;
    }

    virtual Parameter::Builder &trainable(const bool _trainable) {
        assert(!this->_trainable.isPresent());
        this->_trainable = _trainable;
        return *this;
    }

    virtual Parameter::Builder &optimizer(std::shared_ptr<Optimizer> &_optimizerOverride) {
        assert(this->_optimizerOverride == nullptr);
        this->_optimizerOverride = _optimizerOverride;
        return *this;
    }

    virtual Parameter::Builder &optimizer(std::shared_ptr<Optimizer> &&_optimizerOverride) {
        assert(this->_optimizerOverride == nullptr);
        this->_optimizerOverride = _optimizerOverride;
        return *this;
    }

   private:
    std::string _name;
    std::vector<uint64_t> _shape;
    Optional<DataType> _dtype;
    std::shared_ptr<Initializer> _initializer;
    Optional<bool> _trainable;
    std::shared_ptr<Optimizer> _optimizerOverride;
};

}  // namespace Thor
