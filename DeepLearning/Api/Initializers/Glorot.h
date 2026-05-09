#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Initializers/Glorot.h"
#include <optional>


namespace Thor {

class Glorot : public Initializer {
   public:
    class Builder;

    Glorot(ThorImplementation::Glorot::Mode mode) : mode(mode) { initialized = true; }

    ~Glorot() override = default;

    std::shared_ptr<ThorImplementation::Initializer> stamp() override { return ThorImplementation::Glorot(mode).clone(); }

    std::shared_ptr<Initializer> clone() const override { return std::make_shared<Glorot>(*this); }

    nlohmann::json architectureJson() const override;
    static std::shared_ptr<Initializer> deserialize(const nlohmann::json &j);
    ThorImplementation::Glorot::Mode getMode() const { return mode; }

   protected:
    const ThorImplementation::Glorot::Mode mode;
};

class Glorot::Builder : public Initializer::Builder {
   public:
    ~Builder() override = default;

    std::shared_ptr<Initializer> build() override {
        if (!_mode.has_value())
            _mode = ThorImplementation::Glorot::Mode::UNIFORM;

        Glorot glorotInitializer(_mode.value());
        return glorotInitializer.clone();
    }

    virtual Glorot::Builder &mode(ThorImplementation::Glorot::Mode _mode) {
        THOR_THROW_IF_FALSE(!this->_mode.has_value());
        this->_mode = _mode;
        return *this;
    }

    std::shared_ptr<Initializer::Builder> clone() override { return std::make_shared<Glorot::Builder>(*this); }

   protected:
    std::optional<ThorImplementation::Glorot::Mode> _mode;
};

}  // namespace Thor
