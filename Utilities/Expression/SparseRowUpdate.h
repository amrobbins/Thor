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

struct SparseRowUpdateFusionSource;

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
        int deviceNum);

    // Emits the optimizer-owned sparse-row update expression as straight-line source for one reduced float4 lane.
    // The embedding reducer supplies entries listed in localDenseLogicalInputNames directly as local float4 expressions
    // (for example the reduced `gradient` value), so those dense logical tensors are not added to the fused kernel ABI.
    [[nodiscard]] static SparseRowUpdateFusionSource emitFusionSource(
        PhysicalOutputs outputs,
        const Tensor& rows,
        const Tensor& numRows,
        const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& inputs,
        const std::unordered_map<std::string, Tensor>& indexedOutputs,
        const std::unordered_map<std::string, std::string>& localDenseLogicalInputExpressions);

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
        DataType dtype = DataType::FP32;
    };

    struct RuntimeOutputSlot {
        std::string name;
        Tensor tensor;
        DataType dtype = DataType::FP32;
    };

   private:
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string kernelName;
    int deviceNum = 0;
    uint64_t capacity = 0;
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    uint32_t valuesPerThread = 1;
    DataType rowDataType = DataType::UINT64;
    Tensor rows;
    Tensor numRows;
    std::vector<RuntimeInputSlot> inputSlots;
    std::vector<RuntimeOutputSlot> outputSlots;
    std::unordered_map<std::string, Tensor> indexedOutputsByName;
};

struct SparseRowUpdateFusionSource {
    std::string helperSource;
    std::string parameterSource;
    std::string bodySource;
    std::vector<SparseRowUpdatePlan::RuntimeInputSlot> kernelInputSlots;
    std::vector<SparseRowUpdatePlan::RuntimeOutputSlot> outputSlots;
};

}  // namespace ThorImplementation
