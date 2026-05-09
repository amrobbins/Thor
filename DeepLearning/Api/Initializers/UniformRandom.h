#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

#include <chrono>
#include <optional>

namespace Thor {

class UniformRandom : public Initializer {
   public:
    class Builder;

    UniformRandom(float minValue, float maxValue) : minValue(minValue), maxValue(maxValue) { initialized = true; }

    ~UniformRandom() override = default;

    std::shared_ptr<ThorImplementation::Initializer> stamp() override {
        return ThorImplementation::UniformRandom(maxValue, minValue).clone();
    }

    std::shared_ptr<Initializer> clone() const override { return std::make_shared<UniformRandom>(*this); }

    nlohmann::json architectureJson() const override;
    static std::shared_ptr<Initializer> deserialize(const nlohmann::json &j);

    float getMinValue() const { return minValue; }
    float getMaxValue() const { return maxValue; }

   protected:
    const float minValue;
    const float maxValue;
};

class UniformRandom::Builder : public Initializer::Builder {
   public:
    ~Builder() override = default;

    std::shared_ptr<Initializer> build() override {
        THOR_THROW_IF_FALSE(this->_minValue.has_value());
        THOR_THROW_IF_FALSE(this->_maxValue.has_value());
        THOR_THROW_IF_FALSE(_minValue.value() <= _maxValue.value());

        UniformRandom uniformRandomInitializer(_minValue.value(), _maxValue.value());
        return uniformRandomInitializer.clone();
    }

    virtual UniformRandom::Builder &minValue(float _minValue) {
        THOR_THROW_IF_FALSE(!this->_minValue.has_value());
        this->_minValue = _minValue;
        return *this;
    }

    virtual UniformRandom::Builder &maxValue(float _maxValue) {
        THOR_THROW_IF_FALSE(!this->_maxValue.has_value());
        this->_maxValue = _maxValue;
        return *this;
    }

    std::shared_ptr<Initializer::Builder> clone() override { return std::make_shared<UniformRandom::Builder>(*this); }

   protected:
    std::optional<float> _minValue;
    std::optional<float> _maxValue;
};

}  // namespace Thor
