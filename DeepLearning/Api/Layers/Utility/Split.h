#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/SegmentedPrimitivePlacement.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSplit.h"
#include "DeepLearning/Implementation/Layers/Utility/Split.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

/**
 * Split partitions one tensor along an axis into consecutive pieces: piece i holds sizes[i] entries of the axis, in
 * order, and the entries of every other axis. The axis counts the per-example dimensions (the batch is not one of
 * them). Piece i may have its own data type, outputDataTypes[i], the input's by default; it is converted as it moves.
 * Forward is one launch that writes every piece; backward is one launch that concatenates the pieces' gradients into
 * the input's, converted to the input's type, with zeros for a piece whose consumers send no gradient (an integral
 * piece never does). It is the reverse of Concatenate.
 *
 * A RaggedTensor input splits along a trailing value axis (the axis counts the trailing dimensions, the packed values
 * dimension is not one of them); every piece is a RaggedTensor that shares the input's row partition. Only the active
 * packed values are moved, both ways.
 */
class Split : public MultiConnectionLayer {
   public:
    class Builder;

    Split();
    ~Split() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Split>(*this); }

    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    // The pieces, in order, as RaggedTensors sharing the input's partition; empty for a dense input.
    [[nodiscard]] std::vector<RaggedTensor> getRaggedFeatureOutputs() const { return raggedFeatureOutputs; }

    Tensor getFeatureOutput(Tensor inputTensor) const override {
        (void)inputTensor;
        // One input drives several outputs: use getFeatureOutputs().
        THOR_UNREACHABLE();
    }

    std::optional<Tensor> getFeatureOutput() const override {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    // The values input, for a ragged input.
    Tensor getFeatureInput(Tensor outputTensor) const override {
        for (const Tensor &output : featureOutputs)
            if (output == outputTensor) return featureInputs[0];
        THOR_UNREACHABLE();
    }

    // A ragged input has two tensors, its values and its partition; the pieces are driven once both are connected.
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        bool isInput = false;
        for (const Tensor &input : featureInputs) isInput |= input == inputTensor;
        THOR_THROW_IF_FALSE(isInput);
        if (getUseRagged() && numInputConnectionsMade != featureInputs.size()) return {};
        return featureOutputs;
    }

    bool mustConnectAllInputsToDriveOutput() const override { return getUseRagged(); }
    void informThatInputConnectionMade(Tensor inputTensor) override {
        (void)inputTensor;
        numInputConnectionsMade += 1;
        THOR_THROW_IF_FALSE(numInputConnectionsMade <= featureInputs.size());
    }
    void resetGraphTraversalState() override { numInputConnectionsMade = 0; }

    // The ragged partition input is the physical layer's [1] active-count carrier.
    [[nodiscard]] ThorImplementation::RaggedPartitionRequirement getRaggedPartitionRequirementForInput(
        const Tensor &inputTensor) const override {
        if (getUseRagged() && inputTensor == raggedFeatureInput->getOffsets())
            return ThorImplementation::kRaggedTrailingConcatenatePartitionRequirement;
        return Layer::getRaggedPartitionRequirementForInput(inputTensor);
    }

    // The input is connection type 0 (a ragged input's partition is 1), and output i is connection type i of the
    // physical layer.
    int getConnectionType(Tensor connectingTensor) const override {
        for (uint32_t i = 0; i < featureInputs.size(); ++i)
            if (connectingTensor == featureInputs[i]) return static_cast<int>(i);
        for (uint32_t i = 0; i < featureOutputs.size(); ++i)
            if (connectingTensor == featureOutputs[i]) return static_cast<int>(i);
        throw std::runtime_error("Tensor is not connected to this Split layer.");
    }

    std::string getLayerType() const override { return "Split"; }

    [[nodiscard]] uint32_t getAxis() const { return axis; }
    [[nodiscard]] std::vector<uint64_t> getSizes() const { return sizes; }

    // API values always use the per-example logical shape; physical capacity is derived from the execution batch when
    // the network is placed.
    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor &outputTensor) const override {
        (void)outputTensor;
        return false;
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        std::vector<unsigned long> axisElements(sizes.begin(), sizes.end());
        std::vector<ThorImplementation::DataType> outputDataTypes;
        for (const Tensor &output : featureOutputs) outputDataTypes.push_back(output.getDataType());
        if (getUseRagged()) {
            // Stamped from the values or the partition carrier, whichever is connected first; B comes from that
            // already-placed physical driver, never from the logical [M, ...].
            const RaggedTensor &input = raggedFeatureInput.value();
            const bool partitionPort = connectingApiTensor == input.getOffsets();
            THOR_THROW_IF_FALSE(partitionPort || connectingApiTensor == input.getValues());
            const ThorImplementation::RaggedTensorDescriptor placed =
                SegmentedPrimitivePlacementDetail::physicalRaggedDescriptorForStamp(input, drivingLayer, "Split");
            const uint64_t runtimeB = placed.getBatchSize();
            if (runtimeB == 0 || runtimeB > std::numeric_limits<uint32_t>::max() ||
                placed != input.getDescriptorForPlacement(runtimeB))
                throw std::runtime_error("Split driving row-partition descriptor disagrees with placement.");
            const auto physical = SegmentedPrimitivePlacementDetail::physicalDescriptorForStamp(drivingLayer, "Split");
            if (partitionPort) {
                if (physical.getDataType() != input.getOffsetsDataType() || physical.getDimensions() != std::vector<uint64_t>{1})
                    throw std::runtime_error("Split requires a physical [1] active-count carrier.");
            } else if (physical != placed.getValuesDescriptor()) {
                throw std::runtime_error("Split driving values dimensions disagree with placement.");
            }
            // The values tensors carry the packed values dimension first.
            return std::make_shared<ThorImplementation::RaggedSplit>(axis + 1, axisElements, runtimeB, outputDataTypes);
        }
        (void)drivingLayer;
        THOR_THROW_IF_FALSE(featureInputs.size() == 1 && connectingApiTensor == featureInputs[0]);
        // The physical tensors carry the batch first.
        return std::make_shared<ThorImplementation::Split>(axis + 1, axisElements, outputDataTypes);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)tensorPlacement;
        // The pieces, and the input's gradient (a ragged input's values hold M per example).
        const uint64_t perExample = getOutputTensorBytes(1) + featureInputs[0].getTotalSizeInBytes();
        if (batchSize != 0 && perExample > std::numeric_limits<uint64_t>::max() / batchSize)
            throw std::overflow_error("Split memory accounting overflows uint64_t.");
        return perExample * batchSize;
    }

   private:
    uint32_t axis = 0;
    std::vector<uint64_t> sizes;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::vector<RaggedTensor> raggedFeatureOutputs;
    uint32_t numInputConnectionsMade = 0;

    friend class Network;
};

// network, featureInput (a Tensor or a RaggedTensor), axis and sizes are required. sizes are the pieces' extents on the
// axis, two or more, each positive, summing to the input's extent. outputDataTypes, one per piece, is optional; without
// it every piece has the input's type.
class Split::Builder {
   public:
    Builder() {}

    virtual Split build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        if (!_featureInput.has_value() && !_raggedFeatureInput.has_value())
            THOR_THROW_LOGIC_ERROR("Split requires a feature input.");
        if (!_axis.has_value()) THOR_THROW_LOGIC_ERROR("Split requires an axis.");
        if (_sizes.size() < 2) THOR_THROW_LOGIC_ERROR("Split requires two or more sizes.");
        const bool ragged = _raggedFeatureInput.has_value();
        if (ragged && (_raggedFeatureInput->getRaggedRank() != 1 || !_raggedFeatureInput->hasMaxValuesPerRow()))
            THOR_THROW_LOGIC_ERROR("Split takes a RaggedTensor of ragged rank one with a maximum number of values per row.");
        // The axis counts the per-example dimensions, or a ragged input's trailing dimensions; dimensions[0] of a
        // ragged input's values is the packed values dimension, so its axis is one further on.
        const std::vector<uint64_t> dimensions =
            ragged ? _raggedFeatureInput->getValuesDimensions() : _featureInput->getDimensions();
        const uint32_t first = ragged ? 1 : 0;
        if (_axis.value() >= dimensions.size() - first)
            THOR_THROW_LOGIC_ERROR("Split axis " + std::to_string(_axis.value()) + " is out of range for an input of " +
                                   std::to_string(dimensions.size() - first) +
                                   (ragged ? " trailing dimensions." : " dimensions (the batch is not one of them)."));
        const uint32_t valuesAxis = _axis.value() + first;
        uint64_t total = 0;
        for (const uint64_t size : _sizes) {
            if (size == 0) THOR_THROW_LOGIC_ERROR("Split sizes must be positive.");
            if (size > std::numeric_limits<uint64_t>::max() - total) THOR_THROW_LOGIC_ERROR("Split sizes overflow.");
            total += size;
        }
        if (total != dimensions[valuesAxis])
            THOR_THROW_LOGIC_ERROR("Split sizes sum to " + std::to_string(total) + ", the input's extent on axis " +
                                   std::to_string(_axis.value()) + " is " + std::to_string(dimensions[valuesAxis]) + ".");
        if (_outputDataTypes.has_value() && _outputDataTypes->size() != _sizes.size())
            THOR_THROW_LOGIC_ERROR("Split takes one output data type per piece: " + std::to_string(_sizes.size()) +
                                   " sizes, " + std::to_string(_outputDataTypes->size()) + " output data types.");

        Split split;
        split.axis = _axis.value();
        split.sizes = _sizes;
        const Tensor input = ragged ? _raggedFeatureInput->getValues() : _featureInput.value();
        split.featureInputs.push_back(input);
        if (ragged) {
            split.raggedFeatureInput = _raggedFeatureInput;
            split.featureInputs.push_back(_raggedFeatureInput->getOffsets());
        }
        for (size_t i = 0; i < _sizes.size(); ++i) {
            std::vector<uint64_t> pieceDimensions = dimensions;
            pieceDimensions[valuesAxis] = _sizes[i];
            const DataType dataType = _outputDataTypes.has_value() ? _outputDataTypes->at(i) : input.getDataType();
            const Tensor piece(dataType, pieceDimensions);
            split.featureOutputs.push_back(piece);
            split.inputTensorFromOutputTensor[piece] = input;
            if (ragged) split.raggedFeatureOutputs.push_back(_raggedFeatureInput->withValues(piece));
        }
        split.initialized = true;
        split.addToNetwork(_network.value());
        return split;
    }

    virtual Split::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Split::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value() && !this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        THOR_THROW_IF_FALSE(!_featureInput.getDimensions().empty());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual Split::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value() && !this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.getValues().isInitialized());
        this->_raggedFeatureInput = _featureInput;
        return *this;
    }

    virtual Split::Builder &axis(uint32_t _axis) {
        THOR_THROW_IF_FALSE(!this->_axis.has_value());
        this->_axis = _axis;
        return *this;
    }

    virtual Split::Builder &sizes(std::vector<uint64_t> _sizes) {
        THOR_THROW_IF_FALSE(this->_sizes.empty());
        this->_sizes = std::move(_sizes);
        return *this;
    }

    virtual Split::Builder &outputDataTypes(std::vector<DataType> _outputDataTypes) {
        THOR_THROW_IF_FALSE(!this->_outputDataTypes.has_value());
        for (const DataType dataType : _outputDataTypes) THOR_THROW_IF_FALSE(Tensor::dataTypeValid(dataType));
        this->_outputDataTypes = std::move(_outputDataTypes);
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<uint32_t> _axis;
    std::vector<uint64_t> _sizes;
    std::optional<std::vector<DataType>> _outputDataTypes;
};

}  // namespace Thor
