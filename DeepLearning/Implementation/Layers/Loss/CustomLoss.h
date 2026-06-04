#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ThorImplementation {

class CustomLoss : public Loss {
   public:
    CustomLoss(DynamicExpression lossExpression,
               DynamicExpression gradientExpression,
               std::string predictionsName = "predictions",
               std::string labelsName = "labels",
               std::string lossName = "loss",
               std::string gradientName = "predictions_grad",
               DataType lossDataType = DataType::FP32);

    ~CustomLoss() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    std::optional<Tensor> connectToPredictionsInputLayer(Layer* predictionsInputLayer,
                                                         std::optional<Tensor> featureInput,
                                                         Stream stream,
                                                         bool backPropagateError) override;
    std::optional<Tensor> connectToLabelsInputLayer(Layer* labelsLayer, std::optional<Tensor> labels, Stream labelsStream) override;
    void compileImpl() override;
    void cleanup() override;

    void infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels, std::optional<Tensor> predictions, std::optional<Tensor> lossGradient, Stream stream) override;

    std::string getType() override { return "CustomLoss"; }
    bool isGradientFusedIntoDrivingLayer() const { return gradientFusedIntoDrivingLayer; }

   protected:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    TensorMap buildLossInputs() const;
    TensorMap buildLossOutputs() const;
    TensorMap buildGradientOutputs() const;

    void validateExpressionOutputNames(const DynamicExpression& expression,
                                       const std::string& expectedOutputName,
                                       const std::string& what) const;
    std::pair<std::vector<uint64_t>, DataType> inferExpressionOutputDescriptor(const DynamicExpression& expression,
                                                                                const std::string& outputName,
                                                                                const std::string& what) const;

    DynamicExpression lossExpression;
    DynamicExpression gradientExpression;
    std::string predictionsName;
    std::string labelsName;
    std::string lossName;
    std::string gradientName;
    bool gradientFusedIntoDrivingLayer = false;

    void tryFuseGradientIntoDrivingLayer();

    std::shared_ptr<PreparedDynamicExpression> lossPrepared;
    std::shared_ptr<StampedExecutionPlan> lossStamped;
    std::function<void(Stream&)> lossPreRunHook;

    std::shared_ptr<PreparedDynamicExpression> gradientPrepared;
    std::shared_ptr<StampedExecutionPlan> gradientStamped;
    std::function<void(Stream&)> gradientPreRunHook;
};

}  // namespace ThorImplementation
