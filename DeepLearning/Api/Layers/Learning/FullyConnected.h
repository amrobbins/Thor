#pragma once

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

// #ifdef THOR_TESTING
// #include <gtest/gtest_prod.h>
// #endif

#include <assert.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "TrainableLayer.h"

namespace Thor {

class FullyConnected : public TrainableLayer {
   public:
    using ExpressionTransform = std::function<ThorImplementation::Expression(const ThorImplementation::Expression &)>;

    class Builder;

    FullyConnected() {}

    virtual ~FullyConnected() = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<FullyConnected>(*this); }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;

    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);

    nlohmann::json architectureJson() const override;

    static const char *epilogueInputName() { return "__fully_connected_epilogue_input"; }
    static const char *epilogueOutputName() { return "__fully_connected_epilogue_output"; }

    static ThorImplementation::ExpressionDefinition makeEpilogueDefinition(const ExpressionTransform &transform) {
        if (!transform) {
            throw std::invalid_argument("FullyConnected epilogue transform must be callable.");
        }
        ThorImplementation::Expression input = ThorImplementation::Expression::input(epilogueInputName());
        ThorImplementation::Expression output = transform(input);
        ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
            ThorImplementation::Expression::outputs({{epilogueOutputName(), output}}));
        validateEpilogueDefinition(definition);
        return definition;
    }

    static void validateEpilogueDefinition(const ThorImplementation::ExpressionDefinition &definition) {
        definition.validate();
        if (definition.outputs.outputs.size() != 1 || definition.outputs.outputs.front().name != epilogueOutputName()) {
            throw std::invalid_argument("FullyConnected epilogue expression must have exactly one output named " +
                                        std::string(epilogueOutputName()) + ".");
        }
        if (definition.outputs.expr == nullptr || definition.outputs.expr->inputs.size() != 1 ||
            definition.outputs.expr->inputs.front().name != epilogueInputName() ||
            definition.outputs.expr->inputs.front().kind != ThorImplementation::NamedInput::Kind::Tensor) {
            throw std::invalid_argument("FullyConnected epilogue expression must have exactly one tensor input named " +
                                        std::string(epilogueInputName()) + ".");
        }
    }

   protected:
    void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) override {
        std::vector<uint64_t> inputDimensions = inputTensor.getDimensions();
        int gpuNum = stream.getGpuNum();
        assert(!inputDimensions.empty());

        // No matter the incoming shape, the tensor is treated as a one dimensional tensor for fully connected purposes
        // It becomes a matrix when the batch dimension is included.
        assert(inputDimensions.size() >= 1);
        uint64_t numInputFeatures = 1;
        for (uint32_t i = 0; i < inputDimensions.size(); ++i)
            numInputFeatures *= inputDimensions[i];

        if (batchSize > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            numInputFeatures > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            numOutputFeatures > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FullyConnected matrix dimensions exceed the int32 cuBLASLt interface limit.");
        }

        const auto matmulDataTypes =
            cublasLtMatmulDataTypesForFullyConnected(inputTensor.getDataType(), weightsDataType, computeDataType, outputDataType);

        const int batchRows = static_cast<int>(batchSize);
        const int inputFeatures = static_cast<int>(numInputFeatures);
        const int outputFeatures = static_cast<int>(numOutputFeatures);

        // Forward shapes: [batch, in_features] @ [in_features, out_features] -> [batch, out_features].
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(gpuNum,
                                                                                               batchRows,
                                                                                               inputFeatures,
                                                                                               inputFeatures,
                                                                                               outputFeatures,
                                                                                               inputFeatures,
                                                                                               outputFeatures,
                                                                                               outputFeatures,
                                                                                               false,
                                                                                               false,
                                                                                               matmulDataTypes);

        // Backward shapes wrt input: [batch, out_features] @ transpose([in_features, out_features]) -> [batch, in_features].
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(gpuNum,
                                                                                               batchRows,
                                                                                               outputFeatures,
                                                                                               inputFeatures,
                                                                                               outputFeatures,
                                                                                               outputFeatures,
                                                                                               outputFeatures,
                                                                                               inputFeatures,
                                                                                               false,
                                                                                               true,
                                                                                               matmulDataTypes);

        // Backward shapes wrt weights: transpose([batch, in_features]) @ [batch, out_features] -> [in_features, out_features].
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(gpuNum,
                                                                                               batchRows,
                                                                                               inputFeatures,
                                                                                               batchRows,
                                                                                               outputFeatures,
                                                                                               inputFeatures,
                                                                                               outputFeatures,
                                                                                               outputFeatures,
                                                                                               true,
                                                                                               false,
                                                                                               matmulDataTypes);
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent);

    std::string getLayerType() const override { return "FullyConnected"; }

   private:
    uint32_t numOutputFeatures;
    bool hasBias;
    std::shared_ptr<Activation> activation;
    Tensor::DataType weightsDataType;
    Tensor::DataType computeDataType;
    Tensor::DataType outputDataType;

    // FIXME: These should not be part of Thor::FullyConnected, the builder yes, but the builder should
    //        associate these with the parameters
    std::shared_ptr<Initializer> weightsInitializer;
    std::shared_ptr<Initializer> biasesInitializer;
    std::shared_ptr<Optimizer> weightsOptimizer;
    std::shared_ptr<Optimizer> biasesOptimizer;

    Optional<ThorImplementation::ExpressionDefinition> epilogue;

    static bool isFullyConnectedFloatingDataType(Tensor::DataType dataType) {
        switch (dataType) {
            case Tensor::DataType::FP8_E4M3:
            case Tensor::DataType::FP8_E5M2:
            case Tensor::DataType::FP16:
            case Tensor::DataType::BF16:
            case Tensor::DataType::FP32:
                return true;
            default:
                return false;
        }
    }

    static std::string dataTypeName(Tensor::DataType dataType) {
        return ThorImplementation::TensorDescriptor::getElementTypeName(dataType);
    }

    static uint64_t checkedFeatureCount(const std::vector<uint64_t> &dimensions, const std::string &what) {
        if (dimensions.empty()) {
            throw std::invalid_argument("FullyConnected " + what + " must have at least one feature dimension.");
        }

        uint64_t featureCount = 1;
        for (uint64_t dim : dimensions) {
            if (dim == 0) {
                throw std::invalid_argument("FullyConnected " + what + " dimensions must be non-zero.");
            }
            if (featureCount > std::numeric_limits<uint64_t>::max() / dim) {
                throw std::invalid_argument("FullyConnected " + what + " feature count overflows uint64_t.");
            }
            featureCount *= dim;
        }

        if (featureCount > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FullyConnected " + what + " feature count exceeds the int32 cuBLASLt interface limit.");
        }

        return featureCount;
    }

    static void verifyFullyConnectedDataType(Tensor::DataType dataType, const std::string &what) {
        if (!isFullyConnectedFloatingDataType(dataType)) {
            throw std::invalid_argument("FullyConnected " + what + " must be one of fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32. Got " +
                                        dataTypeName(dataType) + ".");
        }
    }

    static cudaDataType_t cublasLtCudaDataTypeForFullyConnected(Tensor::DataType dataType) {
        switch (dataType) {
            case Tensor::DataType::FP32:
                return CUDA_R_32F;
            case Tensor::DataType::BF16:
                return CUDA_R_16BF;
            case Tensor::DataType::FP16:
                return CUDA_R_16F;
            case Tensor::DataType::FP8_E4M3:
                return CUDA_R_8F_E4M3;
            case Tensor::DataType::FP8_E5M2:
                return CUDA_R_8F_E5M2;
            default:
                throw std::invalid_argument("FullyConnected cuBLASLt dtype check does not support " + dataTypeName(dataType) + ".");
        }
    }

    static Optional<cublasComputeType_t> cublasLtComputeTypeForFullyConnected(Tensor::DataType computeDataType) {
        switch (computeDataType) {
            case Tensor::DataType::FP32:
                return CUBLAS_COMPUTE_32F;
            case Tensor::DataType::FP16:
                return CUBLAS_COMPUTE_32F_FAST_16F;
            case Tensor::DataType::BF16:
                return CUBLAS_COMPUTE_32F_FAST_16BF;
            default:
                return Optional<cublasComputeType_t>::empty();
        }
    }

    static void verifyFullyConnectedComputeDataType(Tensor::DataType dataType) {
        if (cublasLtComputeTypeForFullyConnected(dataType).isEmpty()) {
            throw std::invalid_argument(
                "FullyConnected computeDataType must be fp32, fp16, or bf16 for Thor's current cuBLASLt floating GEMM path. Got " +
                dataTypeName(dataType) + ".");
        }
    }

    static bool isSupportedCublasLtMatmulDataTypesForFullyConnected(
        const ThorImplementation::CublasMatrixMultiply::MatmulDataTypes &dataTypes) {
        if (!isFullyConnectedFloatingDataType(dataTypes.A) || !isFullyConnectedFloatingDataType(dataTypes.B) ||
            !isFullyConnectedFloatingDataType(dataTypes.C) || !isFullyConnectedFloatingDataType(dataTypes.D)) {
            return false;
        }

        const Optional<cublasComputeType_t> computeType = cublasLtComputeTypeForFullyConnected(dataTypes.compute);
        if (computeType.isEmpty()) {
            return false;
        }

        const cudaDataType_t ADataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.A);
        const cudaDataType_t BDataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.B);
        const cudaDataType_t CDataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.C);
        const cudaDataType_t DDataType = cublasLtCudaDataTypeForFullyConnected(dataTypes.D);

        return isSupportedCublasLtOperationType(computeType.get(), CUDA_R_32F, ADataType, BDataType, CDataType, DDataType);
    }

    static ThorImplementation::CublasMatrixMultiply::MatmulDataTypes cublasLtMatmulDataTypesForFullyConnected(
        Tensor::DataType inputDataType,
        Tensor::DataType weightsDataType,
        Tensor::DataType computeDataType,
        Tensor::DataType outputDataType) {
        using MatmulDataTypes = ThorImplementation::CublasMatrixMultiply::MatmulDataTypes;

        const MatmulDataTypes directDataTypes{inputDataType, weightsDataType, outputDataType, outputDataType, computeDataType};
        if (isSupportedCublasLtMatmulDataTypesForFullyConnected(directDataTypes)) {
            return directDataTypes;
        }

        // This mirrors EquationCompiler's safe fallback: when cuBLASLt does not expose the requested mixed A/B plan,
        // the expression path can cast the matrix inputs into the resolved output dtype before the matmul stage.
        const MatmulDataTypes outputDataTypes{outputDataType, outputDataType, outputDataType, outputDataType, computeDataType};
        if (isSupportedCublasLtMatmulDataTypesForFullyConnected(outputDataTypes)) {
            return outputDataTypes;
        }

        throw std::invalid_argument("FullyConnected requested dtype plan is unsupported by Thor's cuBLASLt matmul path. input=" +
                                    dataTypeName(inputDataType) + ", weights=" + dataTypeName(weightsDataType) +
                                    ", compute=" + dataTypeName(computeDataType) + ", output=" + dataTypeName(outputDataType) + ".");
    }

    friend class Network;
    friend class Builder;

    // #ifdef THOR_TESTING
    //     FRIEND_TEST(FullyConnectedTest, SerializeProducesExpectedJson);
    // #endif
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual FullyConnected build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(_numOutputFeatures.isPresent());
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer == nullptr)
            _weightsInitializer = Glorot::Builder().build();
        if (_biasInitializer == nullptr)
            _biasInitializer = Glorot::Builder().build();
        if (!_activation && !_activationExplicitlyRemoved)
            _activation = SoftPlus::Builder().build();
        if (_weightsDataType.isEmpty())
            _weightsDataType = _featureInputs[0].getDataType();
        if (_computeDataType.isEmpty())
            _computeDataType = _featureInputs[0].getDataType();
        if (_outputDataType.isEmpty())
            _outputDataType = _featureInputs[0].getDataType();

        verifyConfig();

        FullyConnected fullyConnected;

        fullyConnected.featureInputs = _featureInputs;
        fullyConnected.numOutputFeatures = _numOutputFeatures;

        fullyConnected.hasBias = _hasBias;
        fullyConnected.weightsInitializer = _weightsInitializer->clone();
        fullyConnected.biasesInitializer = _biasInitializer->clone();
        if (_activation != nullptr)
            fullyConnected.activation = _activation;
        fullyConnected.weightsDataType = _weightsDataType;
        fullyConnected.computeDataType = _computeDataType;
        fullyConnected.outputDataType = _outputDataType;

        // When this layer gets a specific optimizer, set it now, otherwise network will attach the network default optimizer to it.
        fullyConnected.weightsOptimizer = _weightsOptimizer;
        fullyConnected.biasesOptimizer = _biasesOptimizer;

        fullyConnected.epilogue = _epilogue;
        fullyConnected.initialized = true;

        for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i)
            fullyConnected.featureOutputs.push_back(Tensor(fullyConnected.outputDataType, {fullyConnected.numOutputFeatures}));
        fullyConnected.addToNetwork(_network.get());

        return fullyConnected;
    }

    virtual FullyConnected::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual FullyConnected::Builder &featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual FullyConnected::Builder &numOutputFeatures(uint32_t _numOutputFeatures) {
        assert(!this->_numOutputFeatures.isPresent());
        this->_numOutputFeatures = _numOutputFeatures;
        return *this;
    }

    virtual FullyConnected::Builder &hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &_weightsInitializer) {
        assert(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &&_weightsInitializer) {
        assert(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &_biasInitializer) {
        assert(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &&_biasInitializer) {
        assert(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &_activation) {
        assert(this->_activation == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &&_activation) {
        assert(this->_activation == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &weightsDataType(Tensor::DataType _weightsDataType) {
        assert(this->_weightsDataType.isEmpty());
        this->_weightsDataType = _weightsDataType;
        return *this;
    }

    virtual FullyConnected::Builder &computeDataType(Tensor::DataType _computeDataType) {
        assert(this->_computeDataType.isEmpty());
        this->_computeDataType = _computeDataType;
        return *this;
    }

    virtual FullyConnected::Builder &outputDataType(Tensor::DataType _outputDataType) {
        assert(this->_outputDataType.isEmpty());
        this->_outputDataType = _outputDataType;
        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        assert(!this->_activation);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    virtual FullyConnected::Builder &epilogue(ExpressionTransform transform) {
        assert(this->_epilogue.isEmpty());
        _epilogue = FullyConnected::makeEpilogueDefinition(transform);
        return *this;
    }

    virtual FullyConnected::Builder &epilogue(ThorImplementation::ExpressionDefinition definition) {
        assert(this->_epilogue.isEmpty());
        FullyConnected::validateEpilogueDefinition(definition);
        _epilogue = std::move(definition);
        return *this;
    }

    virtual FullyConnected::Builder &weightsOptimizer(std::shared_ptr<Optimizer> _weightsOptimizer) {
        assert(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = _weightsOptimizer;
        return *this;
    }

    virtual FullyConnected::Builder &biasesOptimizer(std::shared_ptr<Optimizer> _biasesOptimizer) {
        assert(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = _biasesOptimizer;
        return *this;
    }

   private:
    void verifyConfig() const {
        if (!_network.isPresent()) {
            throw std::invalid_argument("FullyConnected::Builder requires network().");
        }
        if (_featureInputs.empty()) {
            throw std::invalid_argument("FullyConnected::Builder requires at least one featureInput().");
        }
        if (!_numOutputFeatures.isPresent()) {
            throw std::invalid_argument("FullyConnected::Builder requires numOutputFeatures().");
        }
        if (_numOutputFeatures.get() == 0) {
            throw std::invalid_argument("FullyConnected numOutputFeatures must be non-zero.");
        }
        if (_numOutputFeatures.get() > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FullyConnected numOutputFeatures exceeds the int32 cuBLASLt interface limit.");
        }
        if (_weightsInitializer == nullptr) {
            throw std::invalid_argument("FullyConnected weightsInitializer must be non-null.");
        }
        if (_hasBias.get() && _biasInitializer == nullptr) {
            throw std::invalid_argument("FullyConnected biasInitializer must be non-null when hasBias is true.");
        }
        if (!_activationExplicitlyRemoved && _activation == nullptr) {
            throw std::invalid_argument("FullyConnected activation must be non-null unless noActivation() was requested.");
        }
        if (_epilogue.isPresent()) {
            FullyConnected::validateEpilogueDefinition(_epilogue.get());
        }

        const Tensor::DataType inputDataType = _featureInputs.front().getDataType();
        const std::vector<uint64_t> inputDimensions = _featureInputs.front().getDimensions();
        FullyConnected::checkedFeatureCount(inputDimensions, "feature input");
        FullyConnected::verifyFullyConnectedDataType(inputDataType, "feature input data type");
        FullyConnected::verifyFullyConnectedDataType(_weightsDataType.get(), "weightsDataType");
        FullyConnected::verifyFullyConnectedComputeDataType(_computeDataType.get());
        FullyConnected::verifyFullyConnectedDataType(_outputDataType.get(), "outputDataType");

        // Validate the matmul data-type plan against the same cuBLASLt support table used by CublasMatrixMultiply.
        (void)FullyConnected::cublasLtMatmulDataTypesForFullyConnected(
            inputDataType, _weightsDataType.get(), _computeDataType.get(), _outputDataType.get());

        for (uint32_t i = 0; i < _featureInputs.size(); ++i) {
            const Tensor &featureInput = _featureInputs[i];
            if (!featureInput.isInitialized()) {
                throw std::invalid_argument("FullyConnected featureInput " + std::to_string(i) + " is not initialized.");
            }
            if (featureInput.getDataType() != inputDataType) {
                throw std::invalid_argument("FullyConnected all feature inputs must have the same data type.");
            }
            if (featureInput.getDimensions() != inputDimensions) {
                throw std::invalid_argument("FullyConnected all feature inputs must have the same dimensions.");
            }
            FullyConnected::checkedFeatureCount(featureInput.getDimensions(), "feature input " + std::to_string(i));
        }
    }

    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    std::shared_ptr<Activation> _activation;
    Optional<Tensor::DataType> _weightsDataType;
    Optional<Tensor::DataType> _computeDataType;
    Optional<Tensor::DataType> _outputDataType;

    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasInitializer;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
    bool _activationExplicitlyRemoved;

    // FIXME: Future optimization, automatically fuse adjacent epilogue expressions from adjacent layers.
    Optional<ThorImplementation::ExpressionDefinition> _epilogue;
};

}  // namespace Thor
