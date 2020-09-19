#pragma once

namespace Thor {

class DropOut : public LayerBase {
   public:
    class Builder;
    DropOut() : initialized(false) {}

    virtual ~DropOut();

   private:
    bool initialized;
    float dropProportion;
    // FIXME: Add feature input
};

class DropOut::Builder {
   public:
    virtual Layer build() {
        assert(_dropProportion.isPresent());

        DropOut *dropOut = new DropOut();
        dropOut->dropProportion = _dropProportion;
        dropOut->initialized = true;
        return Layer(dropOut);
    }

    DropOut::Builder exponentialRunningAverageFactor(float dropProportion) {
        assert(!_dropProportion.isPresent());
        assert(dropProportion > 0.0);
        this->_dropProportion = dropProportion;
        return *this;
    }

   private:
    Optional<float> _dropProportion;
};

}  // namespace Thor
