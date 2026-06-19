#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

class ParameterConstraint {
   public:
    virtual ~ParameterConstraint() = default;

    virtual void apply(Tensor& parameterStorage, Stream& stream) const = 0;
    [[nodiscard]] virtual bool supportsDenseExpressionFusion() const { return false; }
    [[nodiscard]] virtual Expression applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                                    const std::string& namePrefix) const {
        (void)unconstrainedParameterUpdate;
        (void)namePrefix;
        throw std::runtime_error("ParameterConstraint does not support dense expression fusion.");
    }
    [[nodiscard]] virtual std::shared_ptr<ParameterConstraint> clone() const = 0;
    [[nodiscard]] virtual std::string getConstraintType() const = 0;
};

class NonNegativeParameterConstraint : public ParameterConstraint {
   public:
    void apply(Tensor& parameterStorage, Stream& stream) const override;
    [[nodiscard]] bool supportsDenseExpressionFusion() const override;
    [[nodiscard]] Expression applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                            const std::string& namePrefix) const override;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::string getConstraintType() const override;
};

class NonPositiveParameterConstraint : public ParameterConstraint {
   public:
    void apply(Tensor& parameterStorage, Stream& stream) const override;
    [[nodiscard]] bool supportsDenseExpressionFusion() const override;
    [[nodiscard]] Expression applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                            const std::string& namePrefix) const override;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::string getConstraintType() const override;
};

class MinParameterConstraint : public ParameterConstraint {
   public:
    explicit MinParameterConstraint(double minValue);

    void apply(Tensor& parameterStorage, Stream& stream) const override;
    [[nodiscard]] bool supportsDenseExpressionFusion() const override;
    [[nodiscard]] Expression applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                            const std::string& namePrefix) const override;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::string getConstraintType() const override;
    [[nodiscard]] double getMinValue() const;

   private:
    double minValue;
};

class MaxParameterConstraint : public ParameterConstraint {
   public:
    explicit MaxParameterConstraint(double maxValue);

    void apply(Tensor& parameterStorage, Stream& stream) const override;
    [[nodiscard]] bool supportsDenseExpressionFusion() const override;
    [[nodiscard]] Expression applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                            const std::string& namePrefix) const override;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::string getConstraintType() const override;
    [[nodiscard]] double getMaxValue() const;

   private:
    double maxValue;
};

class MinMaxParameterConstraint : public ParameterConstraint {
   public:
    MinMaxParameterConstraint(double minValue, double maxValue);

    void apply(Tensor& parameterStorage, Stream& stream) const override;
    [[nodiscard]] bool supportsDenseExpressionFusion() const override;
    [[nodiscard]] Expression applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                            const std::string& namePrefix) const override;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::string getConstraintType() const override;
    [[nodiscard]] double getMinValue() const;
    [[nodiscard]] double getMaxValue() const;

   private:
    double minValue;
    double maxValue;
};

}  // namespace ThorImplementation
