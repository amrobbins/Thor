#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/Glorot.h"

#include <assert.h>

namespace Thor {

class Glorot : public Initializer {
   public:
    class Builder;

    virtual ~Glorot() = default;

    virtual void stamp(ThorImplementation::Layer *layerThatOwnsTensor, ThorImplementation::Tensor tensorToInitialize) {
        layerThatOwnsTensor->setInitializer(tensorToInitialize, ThorImplementation::Glorot(mode).clone());
    }

    virtual std::shared_ptr<Initializer> clone() const { return std::make_shared<Glorot>(*this); }

    virtual nlohmann::json serialize() const;
    static std::shared_ptr<Initializer> deserialize(const nlohmann::json &j);
    ThorImplementation::Glorot::Mode getMode() const { return mode; }

   protected:
    ThorImplementation::Glorot::Mode mode;
};

class Glorot::Builder : public Initializer::Builder {
   public:
    virtual ~Builder() = default;

    virtual std::shared_ptr<Initializer> build() {
        if (_mode.isEmpty())
            _mode = ThorImplementation::Glorot::Mode::UNIFORM;

        Glorot glorotInitializer;
        glorotInitializer.mode = _mode.get();
        glorotInitializer.initialized = true;
        return glorotInitializer.clone();
    }

    virtual Glorot::Builder &mode(ThorImplementation::Glorot::Mode _mode) {
        assert(!this->_mode.isPresent());
        this->_mode = _mode;
        return *this;
    }

    virtual std::shared_ptr<Initializer::Builder> clone() { return std::make_shared<Glorot::Builder>(*this); }

   protected:
    Optional<ThorImplementation::Glorot::Mode> _mode;
};

}  // namespace Thor
