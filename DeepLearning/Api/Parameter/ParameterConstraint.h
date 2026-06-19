#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace ThorImplementation {
class ParameterConstraint;
}

namespace Thor {

class ParameterConstraint {
   public:
    virtual ~ParameterConstraint() = default;

    [[nodiscard]] virtual std::shared_ptr<ParameterConstraint> clone() const = 0;
    [[nodiscard]] virtual std::shared_ptr<ThorImplementation::ParameterConstraint> stamp() const = 0;
    [[nodiscard]] virtual std::string getConstraintType() const = 0;
    [[nodiscard]] virtual nlohmann::json architectureJson() const;

    static std::string getVersion();
    static std::shared_ptr<ParameterConstraint> deserialize(const nlohmann::json& j);
};

class NonNegativeParameterConstraint : public ParameterConstraint {
   public:
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::shared_ptr<ThorImplementation::ParameterConstraint> stamp() const override;
    [[nodiscard]] std::string getConstraintType() const override;
};

class NonPositiveParameterConstraint : public ParameterConstraint {
   public:
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::shared_ptr<ThorImplementation::ParameterConstraint> stamp() const override;
    [[nodiscard]] std::string getConstraintType() const override;
};

class MinParameterConstraint : public ParameterConstraint {
   public:
    explicit MinParameterConstraint(double minValue);

    [[nodiscard]] double getMinValue() const;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::shared_ptr<ThorImplementation::ParameterConstraint> stamp() const override;
    [[nodiscard]] std::string getConstraintType() const override;
    [[nodiscard]] nlohmann::json architectureJson() const override;

   private:
    double minValue;
};

class MaxParameterConstraint : public ParameterConstraint {
   public:
    explicit MaxParameterConstraint(double maxValue);

    [[nodiscard]] double getMaxValue() const;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::shared_ptr<ThorImplementation::ParameterConstraint> stamp() const override;
    [[nodiscard]] std::string getConstraintType() const override;
    [[nodiscard]] nlohmann::json architectureJson() const override;

   private:
    double maxValue;
};

class MinMaxParameterConstraint : public ParameterConstraint {
   public:
    MinMaxParameterConstraint(double minValue, double maxValue);

    [[nodiscard]] double getMinValue() const;
    [[nodiscard]] double getMaxValue() const;
    [[nodiscard]] std::shared_ptr<ParameterConstraint> clone() const override;
    [[nodiscard]] std::shared_ptr<ThorImplementation::ParameterConstraint> stamp() const override;
    [[nodiscard]] std::string getConstraintType() const override;
    [[nodiscard]] nlohmann::json architectureJson() const override;

   private:
    double minValue;
    double maxValue;
};

}  // namespace Thor
