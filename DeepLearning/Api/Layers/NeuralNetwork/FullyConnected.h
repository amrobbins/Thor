/*
#include "DeepLearning/Implementation/Initializers/Initializer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Layer.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/TensorCoreMatrixMultiply.h"

#include <assert.h>

#include <unordered_map>

using std::unordered_map;

class FullyConnected : public Layer {
   public:
    class Builder;

    virtual Tensor connect(Tensor inputTensor);

    virtual Tensor getWeights();

   private:
    FullyConnected(TensorDescriptor inputDescriptor,
                   unsigned int outputWidth,
                   Initializer weightsInitializer,
                   bool useBias,
                   Initializer biasInitializer,
                   Activation activation);

    unsigned int outputWidth;
    Initializer weightsInitializer;
    bool useBias;
    Initializer biasInitializer;
    Activation activation;
    TensorDescriptor inputDescriptor;

    Tensor weights;
    bool useWorkspace;
    Tensor workspace;

    int numInputFeatures;
    int batchSize

    // allocate anything needed for execution, choose optimal kernels, etc.
    virtual void compile(TensorPlacement placement);

    // initialize weights using the configured initializer. In general set any initial values.
    virtual void initialize();

    // Forward pass
    virtual void infer(Tensor inputTensor, Tensor outputTensor);

    // Backward pass
    // errorIn is passed backward from the next later layer to this layer.
    // errorOut is passed backward from this layer to the previous layer.
    virtual void backProp(Optional<Tensor> errorIn, Optional<Tensor> errorOut, Tensor weightsGradient);
};

class FullyConnected::Builder {
   public:
    Builder();
    virtual FullyConnected build();

    void inputDescriptor(TensorDescriptor inputDescriptor);
    void outputWidth(unsigned int width);
    void initializer(Initializer initializer);
    void useBias(bool useBias);
    void biasInitializer(Initializer initializer);
    void activation(Activation activation);

   private:
    bool outputWidthSet;
    bool weightsInitializerSet;
    bool useBiasSet;
    bool biasInitializerSet;
    bool activationSet;
    bool inputDescriptorSet;

    unsigned int outputWidth;
    Initializer weightsInitializer;
    bool useBias;
    Initializer biasInitializer;
    Activation activation;
    TensorDescriptor inputDescriptor;
    TensorPlacement layerPlacement;
};*/
