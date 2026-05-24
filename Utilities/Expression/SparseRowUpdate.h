#pragma once

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"
#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

enum class SparseRowUpdateTensorKind {
    DenseLogicalRows,
    IndexedRows,
};

struct SparseRowUpdateTensorBinding {
    Tensor tensor;
    SparseRowUpdateTensorKind kind = SparseRowUpdateTensorKind::DenseLogicalRows;
};

struct CapturedSparseRowUpdate {
    CapturedSparseRowUpdate() = default;
    explicit CapturedSparseRowUpdate(int deviceNum) : targetNodeHandle(deviceNum) {}

    CapturedSparseRowUpdate(const CapturedSparseRowUpdate&) = delete;
    CapturedSparseRowUpdate& operator=(const CapturedSparseRowUpdate&) = delete;
    CapturedSparseRowUpdate(CapturedSparseRowUpdate&&) noexcept = default;
    CapturedSparseRowUpdate& operator=(CapturedSparseRowUpdate&&) noexcept = default;

    void uploadTargetNode(Stream stream) const { targetNodeHandle.upload(targetNode, stream); }

    DeviceUpdatableKernelNodeDeviceHandle targetNodeHandle;
    DeviceUpdatableKernelNode targetNode;
};

class SparseRowUpdatePlan {
   public:
    SparseRowUpdatePlan() = default;
    SparseRowUpdatePlan(const SparseRowUpdatePlan&) = delete;
    SparseRowUpdatePlan& operator=(const SparseRowUpdatePlan&) = delete;
    SparseRowUpdatePlan(SparseRowUpdatePlan&& other) noexcept;
    SparseRowUpdatePlan& operator=(SparseRowUpdatePlan&& other) noexcept;
    ~SparseRowUpdatePlan();

    [[nodiscard]] static std::unique_ptr<SparseRowUpdatePlan> compile(
        PhysicalOutputs outputs,
        const Tensor& rows,
        const Tensor& numRows,
        const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& inputs,
        const std::unordered_map<std::string, Tensor>& indexedOutputs,
        int deviceNum,
        bool useFastMath = true);

    void run(const std::unordered_map<std::string, float>& runtimeScalars, Stream stream) const;
    void capture(CudaGraphCaptureBuilder& builder,
                 CapturedSparseRowUpdate& captured,
                 const std::unordered_map<std::string, float>& runtimeScalars) const;

    [[nodiscard]] std::vector<std::string> outputNames() const;
    [[nodiscard]] std::unordered_map<std::string, Tensor> getFinalOutputs() const { return indexedOutputsByName; }

   public:
    struct RuntimeInputSlot {
        std::string name;
        NamedInput::Kind inputKind = NamedInput::Kind::Tensor;
        SparseRowUpdateTensorKind tensorKind = SparseRowUpdateTensorKind::DenseLogicalRows;
        Tensor tensor;
        TensorDescriptor::DataType dtype = TensorDescriptor::DataType::FP32;
    };

    struct RuntimeOutputSlot {
        std::string name;
        Tensor tensor;
        TensorDescriptor::DataType dtype = TensorDescriptor::DataType::FP32;
    };

   private:
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string kernelName;
    int deviceNum = 0;
    uint64_t capacity = 0;
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    TensorDescriptor::DataType rowDataType = TensorDescriptor::DataType::UINT64;
    Tensor rows;
    Tensor numRows;
    std::vector<RuntimeInputSlot> inputSlots;
    std::vector<RuntimeOutputSlot> outputSlots;
    std::unordered_map<std::string, Tensor> indexedOutputsByName;
};

}  // namespace ThorImplementation
