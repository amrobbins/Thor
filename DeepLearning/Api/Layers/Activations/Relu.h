#pragma once

namespace Thor {

class Activation;

class Relu : public ActivationBase {
   public:
    class Builder;
    Relu() : initialized(false) {}

    virtual ~Relu();

   private:
    bool initialized;
    // FIXME: Add feature input
};

class Relu::Builder {
   public:
    virtual Activation build() {
        Relu *relu = new Relu();
        relu->initialized = true;
        return Activation(relu);
    }

   private:
    Optional<float> _dropProportion;
};

}  // namespace Thor
