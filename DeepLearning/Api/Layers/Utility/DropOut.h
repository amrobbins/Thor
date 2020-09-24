#pragma once

namespace Thor {

class DropOut : public Layer {
   public:
    class Builder;
    DropOut() : initialized(false) {}

    virtual ~DropOut() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<DropOut>(*this); }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement, uint32_t batchSize) const {
        // FIXME
        return nullptr;
    }

   private:
    float dropProportion;
    bool initialized;
};

class DropOut::Builder {
   public:
    virtual DropOut build() {
        assert(_featureInput.isPresent());
        assert(_dropProportion.isPresent());

        DropOut dropOut;
        dropOut.featureInput = _featureInput;
        dropOut.featureOutput = _featureInput.get().clone();
        dropOut.dropProportion = _dropProportion;
        dropOut.initialized = true;
        return dropOut;
    }

    virtual DropOut::Builder &featureInput(Tensor _featureInput) {
        assert(!this->_featureInput.isPresent());
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual DropOut::Builder &dropProportion(float dropProportion) {
        assert(!_dropProportion.isPresent());
        assert(dropProportion > 0.0);
        this->_dropProportion = dropProportion;
        return *this;
    }

   private:
    Optional<Tensor> _featureInput;
    Optional<float> _dropProportion;
};

}  // namespace Thor
