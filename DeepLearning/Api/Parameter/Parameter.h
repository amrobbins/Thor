#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Common/Optional.h"

namespace Thor {
class Optimizer;
class Parameterizable;

class Parameter {
   public:
    using DataType = ThorImplementation::TensorDescriptor::DataType;
    class Builder;

    virtual ~Parameter() = default;

    static std::string getVersion();
    virtual nlohmann::json architectureJson() const;
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork &stampedNetwork) const;
    static Parameter deserialize(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader);

    bool isTrainable() const;
    bool isTrainingEnabled() const;
    void setTrainingEnabled(bool enabled);

    bool hasOptimizer() const;
    std::shared_ptr<Optimizer> getOptimizer();

   private:
    bool initialized = false;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    Optional<std::string> storageFile;

    std::string name{};
    std::vector<uint64_t> shape{};
    DataType dtype = DataType::FP32;
    std::shared_ptr<Initializer> initializer = nullptr;
    bool trainable = false;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    bool trainingEnabled = false;

    std::shared_ptr<Parameterizable> owner = nullptr;

    Tensor storage;
};

class Parameter::Builder {
   public:
    virtual ~Builder() = default;
    virtual std::shared_ptr<Parameter> build();

    virtual Parameter::Builder &name(const std::string &_name);
    virtual Parameter::Builder &shape(const std::vector<uint64_t> &_shape);
    virtual Parameter::Builder &dtype(const DataType &_dtype);
    virtual Parameter::Builder &initializer(std::shared_ptr<Initializer> &_initializer);
    virtual Parameter::Builder &initializer(std::shared_ptr<Initializer> &&_initializer);
    virtual Parameter::Builder &trainable(const bool _trainable);
    virtual Parameter::Builder &optimizer(std::shared_ptr<Optimizer> &_optimizerOverride);
    virtual Parameter::Builder &optimizer(std::shared_ptr<Optimizer> &&_optimizerOverride);
    virtual Parameter::Builder &owner(const std::shared_ptr<Parameterizable> &_owner);

   private:
    std::string _name{};
    std::vector<uint64_t> _shape{};
    Optional<DataType> _dtype = Optional<DataType>::empty();
    std::shared_ptr<Initializer> _initializer = nullptr;
    Optional<bool> _trainable = Optional<bool>::empty();
    std::shared_ptr<Optimizer> _optimizerOverride = nullptr;
    std::shared_ptr<Parameterizable> _owner = nullptr;
};

}  // namespace Thor
