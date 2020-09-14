#pragma once

namespace Thor {

class Tanh : public ActivationBase {
   public:
    class Builder;
    Tanh() : initialized(false) {}

    virtual ~Tanh();

   private:
    bool initialized;
    float dropProportion;
};

class Tanh::Builder {
   public:
    virtual Layer build() {
        Tanh *tanh = new Tanh();
        tanh->initialized = true;
        return Layer(tanh);
    }

   private:
    Optional<float> _dropProportion;
};

}  // namespace Thor
