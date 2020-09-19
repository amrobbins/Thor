#pragma once

namespace Thor {

class Activation;

class Tanh : public ActivationBase {
   public:
    class Builder;
    Tanh() : initialized(false) {}

    virtual ~Tanh();

   private:
    bool initialized;
    // FIXME: Add feature input
};

class Tanh::Builder {
   public:
    virtual Activation build() {
        Tanh *tanh = new Tanh();
        tanh->initialized = true;
        return Activation(tanh);
    }

   private:
    Optional<float> _dropProportion;
};

}  // namespace Thor
