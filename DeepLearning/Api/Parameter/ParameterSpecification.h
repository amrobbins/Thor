#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/Common/Optional.h"

namespace Thor {
class Optimizer;
class BoundParameter;
class Parameterizable;

class ParameterSpecification {
   public:
    using DataType = ThorImplementation::TensorDescriptor::DataType;
    using StorageContext = ThorImplementation::PhysicalParameter::StorageContext;
    using StorageContextStorageFactory = std::function<ThorImplementation::Tensor(const StorageContext&)>;
    class Builder;

    ParameterSpecification() = default;
    ParameterSpecification(std::string name,
                           const std::vector<uint64_t>& shape,
                           DataType dtype = DataType::FP32,
                           std::shared_ptr<Initializer> initializer = nullptr,
                           bool trainable = true,
                           std::shared_ptr<Optimizer> optimizer = nullptr,
                           bool trainingInitiallyEnabled = true);
    ParameterSpecification(std::string name,
                           StorageContextStorageFactory createStorage,
                           std::shared_ptr<Initializer> initializer = nullptr,
                           bool trainable = true,
                           std::shared_ptr<Optimizer> optimizer = nullptr,
                           bool trainingInitiallyEnabled = true);

    virtual ~ParameterSpecification() = default;

    static std::string getVersion();
    virtual nlohmann::json architectureJson() const;
    virtual nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork& stampedNetwork) const;
    static ParameterSpecification deserialize(const nlohmann::json& j, std::shared_ptr<thor_file::TarReader>& archiveReader);

    [[nodiscard]] const std::string& getName() const;
    [[nodiscard]] std::shared_ptr<Initializer> getInitializer() const;

    bool isTrainable() const;
    bool isTrainingInitiallyEnabled() const;

    bool hasOptimizer() const;
    std::shared_ptr<Optimizer> getOptimizer();

    // Convenience helper to allocate storage with the same properties (placement, etc.) as inputTensor,
    // but having shape and dtype.
    static ThorImplementation::Tensor allocateStorage(const ThorImplementation::Tensor& inputTensor,
                                                      const std::vector<uint64_t>& shape,
                                                      DataType dtype);

    // Build an implementation parameter that delegates storage creation to the factory bound on this API parameter.
    virtual std::shared_ptr<ThorImplementation::PhysicalParameter> stamp();

   private:
    static void validateShape(const std::vector<uint64_t>& shape);
    void validateReadyForUse() const;
    void validateStorageFactoryReadyForStamping() const;

    bool initialized = false;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    Optional<std::string> storageFile;

    std::string name{};
    std::shared_ptr<Initializer> initializer = nullptr;
    bool trainable = false;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    bool trainingInitiallyEnabled = true;

    // A parameter uses either storageContextCreateStorage function to determine attributes at layer compile time
    // -- OR --
    // shape and dtype to determine attributes at parameter definition time
    StorageContextStorageFactory storageContextCreateStorage = nullptr;
    Optional<std::vector<uint64_t>> shape = Optional<std::vector<uint64_t>>::empty();
    Optional<DataType> dtype = Optional<DataType>::empty();

    Tensor storage;

    friend class BoundParameter;
};

class ParameterSpecification::Builder {
   public:
    virtual ~Builder() = default;
    virtual std::shared_ptr<ParameterSpecification> build();

    virtual ParameterSpecification::Builder& name(const std::string& _name);
    virtual ParameterSpecification::Builder& initializer(std::shared_ptr<Initializer>& _initializer);
    virtual ParameterSpecification::Builder& initializer(std::shared_ptr<Initializer>&& _initializer);
    virtual ParameterSpecification::Builder& trainable(const bool _trainable);
    virtual ParameterSpecification::Builder& optimizer(std::shared_ptr<Optimizer>& _optimizerOverride);
    virtual ParameterSpecification::Builder& optimizer(std::shared_ptr<Optimizer>&& _optimizerOverride);
    virtual ParameterSpecification::Builder& createStorage(StorageContextStorageFactory createStorage);

   private:
    std::string _name{};
    std::shared_ptr<Initializer> _initializer = nullptr;
    Optional<bool> _trainable = Optional<bool>::empty();
    std::shared_ptr<Optimizer> _optimizerOverride = nullptr;
    StorageContextStorageFactory _storageContextCreateStorage = nullptr;
};

}  // namespace Thor
