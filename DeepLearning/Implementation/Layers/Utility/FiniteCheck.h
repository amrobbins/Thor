#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Utility/FiniteCheckKernel.h"

#include <cstdint>
#include <optional>
#include <string>

namespace ThorImplementation {

class FiniteCheck : public Layer {
   public:
    FiniteCheck(std::string tensorLabel,
                uint64_t apiTensorId,
                uint64_t originalApiTensorId,
                bool checkForward,
                bool checkBackward,
                bool failOnNonFinite,
                uint32_t maxReportedIndices);
    ~FiniteCheck() override;

    std::optional<Tensor> createFeatureOutputTensor() override;
    void connectToNextLayer(Layer *nextLayer, int driverConnectionType = 0, int loaderConnectionType = 0) override;
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
    FiniteCheckResult checkCpuTensor(const Tensor &tensor) const;
    std::string formatFailure(const Tensor &tensor,
                              const char *direction,
                              const char *tensorRole,
                              const FiniteCheckResult &result) const;

    std::string tensorLabel;
    uint64_t apiTensorId;
    uint64_t originalApiTensorId;
    bool checkForward;
    bool checkBackward;
    bool failOnNonFinite;
    uint32_t maxReportedIndices;
    FiniteCheckResult *gpuResult = nullptr;
};

}  // namespace ThorImplementation
