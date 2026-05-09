#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/UniformRandom.h"

#include <chrono>

namespace Thor {

class UniformRandom : public Initializer {
   public:
    class Builder;

    UniformRandom(float minValue, float maxValue) : minValue(minValue), maxValue(maxValue) { initialized = true; }

    virtual ~UniformRandom() = default;

    std::shared_ptr<ThorImplementation::Initializer> stamp() override {
        return ThorImplementation::UniformRandom(maxValue, minValue).clone();
    }

    virtual std::shared_ptr<Initializer> clone() const { return std::make_shared<UniformRandom>(*this); }

    virtual nlohmann::json architectureJson() const;
    static std::shared_ptr<Initializer> deserialize(const nlohmann::json &j);

    float getMinValue() const { return minValue; }
    float getMaxValue() const { return maxValue; }

   protected:
    const float minValue;
    const float maxValue;
};

class UniformRandom::Builder : public Initializer::Builder {
   public:
    virtual ~Builder() = default;

    virtual std::shared_ptr<Initializer> build() {
        THOR_THROW_IF_FALSE(this->_minValue.isPresent());
        THOR_THROW_IF_FALSE(this->_maxValue.isPresent());
        THOR_THROW_IF_FALSE(_minValue.get() <= _maxValue.get());

        UniformRandom uniformRandomInitializer(_minValue, _maxValue);
        return uniformRandomInitializer.clone();
    }

    virtual UniformRandom::Builder &minValue(float _minValue) {
        THOR_THROW_IF_FALSE(!this->_minValue.isPresent());
        this->_minValue = _minValue;
        return *this;
    }

    virtual UniformRandom::Builder &maxValue(float _maxValue) {
        THOR_THROW_IF_FALSE(!this->_maxValue.isPresent());
        this->_maxValue = _maxValue;
        return *this;
    }

    virtual std::shared_ptr<Initializer::Builder> clone() { return std::make_shared<UniformRandom::Builder>(*this); }

   protected:
    Optional<float> _minValue;
    Optional<float> _maxValue;
};

}  // namespace Thor
