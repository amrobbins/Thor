#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Utility/FiniteCheckKernel.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace ThorImplementation {

class FiniteCheck : public Layer {
   public:
    struct RaggedConfiguration {
        uint64_t batchSize = 0;
        uint64_t maxTotalValues = 0;
        uint64_t elementsPerValue = 1;
        DataType offsetsDataType = DataType::UINT32;
    };

    FiniteCheck(std::string tensorLabel,
                uint64_t apiTensorId,
                uint64_t originalApiTensorId,
                bool checkForward,
                bool checkBackward,
                bool failOnNonFinite,
                uint32_t maxReportedIndices,
                bool enabled,
                std::optional<RaggedConfiguration> raggedConfiguration = std::nullopt);
    ~FiniteCheck() override;

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> connectToPreviousLayer(
        Layer *previousLayer,
        std::optional<Tensor> connectedInput,
        Stream connectedStream,
        bool backPropagateError,
        int connectionType = 0) override;
    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override;
    void forward(std::optional<Tensor> arrivingInput, bool validationPass, uint32_t batchSize = 0) override;
    void cleanup() override;

    std::string getType() override { return "FiniteCheck"; }

   protected:
    void compileImpl() override;
    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;
    void backProp(std::optional<Tensor> dataIn,
                  std::optional<Tensor> errorIn,
                  std::optional<Tensor> errorOut,
                  Stream stream) override;

   private:
    void fuseBackwardAlias();
    void checkTensor(const Tensor &tensor, const char *direction, const char *tensorRole, Stream stream);
    void checkRaggedTensor(const Tensor &tensor, const char *direction, const char *tensorRole, Stream stream);
    FiniteCheckResult checkCpuTensor(const Tensor &tensor, uint64_t numElements) const;
    std::string formatFailure(const Tensor &tensor,
                              const char *direction,
                              const char *tensorRole,
                              const FiniteCheckResult &result) const;
    void validateRaggedInputs() const;
    void resetRaggedForwardArrivalState();

    std::string tensorLabel;
    uint64_t apiTensorId;
    uint64_t originalApiTensorId;
    bool checkForward;
    bool enabled;
    bool checkBackward;
    bool failOnNonFinite;
    uint32_t maxReportedIndices;
    FiniteCheckResult *gpuResult = nullptr;
    std::optional<RaggedConfiguration> raggedConfiguration;
    std::optional<Tensor> rowPartitionInput;
    Stream rowPartitionStream;
    Event rowPartitionReadyEvent;
    bool raggedValuesArrived = false;
    bool raggedPartitionArrived = false;
    std::optional<bool> pendingRaggedValidationPass;
    std::optional<uint32_t> pendingRaggedBatchSize;

    // Normal execution serializes host submission per stamp. Keep the diagnostic
    // workspace/result one-at-a-time defensively without imposing a mutex on every Layer.
    std::mutex checkMutex;
};

}  // namespace ThorImplementation
