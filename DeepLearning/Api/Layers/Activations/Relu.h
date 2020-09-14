#pragma once

namespace Thor {

class Relu : public ActivationBase {
   public:
    class Builder;
    Relu() : initialized(false) {}

    virtual ~Relu();

   private:
    bool initialized;
    float dropProportion;
};

class Relu::Builder {
   public:
    virtual Layer build() {
        Relu *relu = new Relu();
        relu->initialized = true;
        return Layer(relu);
    }

   private:
    Optional<float> _dropProportion;
};

}  // namespace Thor
