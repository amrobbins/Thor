#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "Utilities/Common/Optional.h"

namespace Thor {
class Optimizer;
class Parameterizable;

class Parameter {
   public:
    using DataType = ThorImplementation::TensorDescriptor::DataType;
    class Builder;

    Parameter() = default;
    Parameter(std::string name,
              std::vector<uint64_t> shape,
              DataType dtype = DataType::FP32,
              std::shared_ptr<Initializer> initializer = nullptr,
              bool trainable = true,
              std::shared_ptr<Optimizer> optimizer = nullptr);

    virtual ~Parameter() = default;

    static std::string getVersion();
    virtual nlohmann::json architectureJson() const;
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork &stampedNetwork) const;
    static Parameter deserialize(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader);

    [[nodiscard]] const std::string &getName() const;
    [[nodiscard]] const std::vector<uint64_t> &getShape() const;
    [[nodiscard]] DataType getDataType() const;
    [[nodiscard]] std::shared_ptr<Initializer> getInitializer() const;

    bool isTrainable() const;
    bool isTrainingEnabled() const;
    void setTrainingEnabled(bool enabled);

    bool hasOptimizer() const;
    std::shared_ptr<Optimizer> getOptimizer();

    using StorageContext = ThorImplementation::Parameter::StorageContext;

    // API-side parameter storage definition. This runs at physical layer compile time.
    // The default implementation is the basic fixed-shape parameter: allocate this parameter's
    // shape/dtype on the same placement as the context's single feature input. Python subclasses
    // should override create_storage(...) and return a thor.physical.PhysicalTensor.
    virtual ThorImplementation::Tensor createStorage(const StorageContext &context) const;
    virtual ThorImplementation::Tensor createStorage(const ThorImplementation::Tensor &inputTensor) const;
    ThorImplementation::Tensor createStorage(const StorageContext &context, const std::vector<uint64_t> &shape, DataType dtype) const;
    ThorImplementation::Tensor createStorage(const ThorImplementation::Tensor &inputTensor,
                                             const std::vector<uint64_t> &shape,
                                             DataType dtype) const;
    virtual ThorImplementation::Tensor create_storage(const StorageContext &context) const;
    virtual ThorImplementation::Tensor create_storage(const ThorImplementation::Tensor &inputTensor) const;

    // Build an implementation parameter that delegates storage creation back to this API parameter.
    virtual std::shared_ptr<ThorImplementation::Parameter> stamp();

   private:
    static void validateShape(const std::vector<uint64_t> &shape);
    void validateReadyForUse() const;

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

   private:
    std::string _name{};
    std::vector<uint64_t> _shape{};
    Optional<DataType> _dtype = Optional<DataType>::empty();
    std::shared_ptr<Initializer> _initializer = nullptr;
    Optional<bool> _trainable = Optional<bool>::empty();
    std::shared_ptr<Optimizer> _optimizerOverride = nullptr;
};

}  // namespace Thor
