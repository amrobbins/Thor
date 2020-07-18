/*#include "FullyConnected.h"

#include "MLDev.h"

FullyConnected::FullyConnected(TensorDescriptor inputDescriptor,
                               unsigned int outputWidth,
                               Initializer weightsInitializer,
                               bool useBias,
                               Initializer biasInitializer,
                               Activation activation) {
    this->inputDescriptor = inputDescriptor;
    this->outputWidth = outputWidth;
    this->weightsInitializer = weightsInitializer;
    this->useBias = useBias;
    this->biasInitializer = biasInitializer;
    this->activation = activation;
}

Tensor FullyConnected::connect(Tensor inputTensor) {
    if (inputTensor.getDescriptor().getNumDimensions() > 1) {
        Tensor flatTensor = Flatten().connect(inputTensor);
        Layer::connect(flatTensor);
    }
    Layer::connect(inputTensor);
}



// allocate anything needed for execution, choose optimal kernels, etc.
void compile(TensorPlacement placement) {
    assert(placement.getMemDevice() == TensorPlacement::MemDevices::GPU);
    layerPlacement = placement;


    //-----------------------------------------------
    //
    // A = | WFIn0FOut0  WFIn1FOut0  WFIn2FOut0 ... |
    //     | WFIn0FOut1  WFIn1FOut1  WFIn2FOut1 ... |
    //     | WFIn0FOut2  WFIn1FOut2  WFIn2FOut2 ... |
    //
    // B = | FIn0B0      FIn0B1      FIn0B2     ... |
    //     | FIn1B0      FIn1B1      FIn1B2     ... |
    //     | FIn2B0      FIn2B1      FIn2B2     ... |
    //
    // i.e. WeightFor_feature0_output0, feature0_fromBatch0
    //
    // C = AB
    //
    // Dimensions:
    // A: A_rows x A_cols
    // B: A_cols x B_cols
    // C: A_rows x B_cols
    //
    // All data is regular C++ row major
    //
    // ld_A means leading dimension of matrix A,
    // i.e. the number of elements that separate the start
    // of each row of A in memory, usually ld_A = A_cols.
    //
    // A_rows (M): number of outputs
    // A_cols (K): number of input features
    // B_cols (N): batch size
    //-----------------------------------------------
    vector<unsigned long> inputDimensions = inputDescriptor.getNumDimensions();
    batchSize = (int)(inputDimensions[1]);
    numInputFeatures = (int)(inputDimensions[0]);

    int rowsA = outputWidth;
    int colsA = numInputFeatures;
    int colsB = batchSize;
    TensorCoreMatrixMultiply::instance().chooseOptimalKernel(layerPlacement.getDeviceNum(), rowsA, colsA, colsB);
    int workspaceSize = getWorkspaceSizeInBytes(layerPlacement.getDeviceNum(), rowsA, colsA, colsB);
    useWorkspace = workspaceSize > 0;

    vector<unsigned long> weightsDimensions;
    weightsDimensions.push_back(outputWidth);
    weightsDimensions.push_back(numInputFeatures);
    TensorDescriptor weightsDescriptor(inputDescriptor.getDataType(), weightsDimensions);
    weights = Tensor(layerPlacement, weightsDescriptor);
    if(useWorkspace) {
        vector<unsigned long> workspaceDimensions;
        assert(workspaceSize % TensorDescriptor::getArraySize(1, inputDescriptor.getDataType()) == 0);
        workspaceDimensions.push_back(workspaceSize / TensorDescriptor::getArraySize(1, inputDescriptor.getDataType()));
        TensorDescriptor workspaceDescriptor(inputDescriptor.getDataType(), weightsDimensions);
        workspace = Tensor(layerPlacement, workspaceDescriptor);
    }
}

// initialize weights using the configured initializer. In general set any initial values.
void initialize() {}


Tensor infer(Tensor inputTensor, Stream stream) {
    multiply(weights.getMemPtr(),
             inputTensor.getMemPtr(),
             outputTensor.getMemPtr(),
             workspace.getMemPtr(),
             outputWidth,
             numInputFeatures,
             batchSize,
             stream);
}

void backProp(Tensor errorIn, Tensor errorOut, Tensor weightsGradient) {}

Tensor getWeights() { return weights; }

void infer(Tensor inputTensor, Tensor outputTensor) { return; }
void backProp(Tensor errorIn, Tensor errorOut, Tensor weightsGradient) { return; }




//------------------------------------------------------
//
// Builder
//
//------------------------------------------------------
FullyConnected::Builder::Builder() {
    inputDescriptorSet = false;
    outputWidthSet = false;
    weightsInitializerSet = false;
    useBiasSet = false;
    biasInitializerSet = false;
    activationSet = false;
}

void FullyConnected::Builder::inputDescriptor(TensorDescriptor inputDescriptor) {
    inputDescriptorSet = true;
    _inputDescriptor = inputDescriptor;
}


void FullyConnected::Builder::outputWidth(unsigned int width);
void FullyConnected::Builder::initializer(Initializer initializer);
void FullyConnected::Builder::useBias(bool useBias);
void FullyConnected::Builder::biasInitializer(Initializer initializer);
void FullyConnected::Builder::activation(Activation activation);

FullyConnected FullyConnected::Builder::build() {
    assert(inputDescriptorSet == true);
    assert(outputWidthSet == true);

    // FIXME: other assertions
    assert(inputDescriptor.getNumDimensions() == 2);
    vector<unsigned long> inputDimensions = inputDescriptor.getNumDimensions();
    assert(inputDimensions[0] > 0 && inputDimensions[1] > 0);
    assert(inputDescriptor.getNumDimensions() == 2);
    assert(outputWidth > 0);

    if (!weightsInitializerSet)
        weightsInitializer = Initializer();  // fixme glorot_uniform
    if (!useBiasSet)
        useBias = true;
    if (!biasInitializerSet)
        biasInitializer = Initializer();  // fixme zeros
    if (!activationSet)
        activation = Activation();  // fixme relu
    if (!inputDescriptorSet)
        inputDescriptor = TensorDescriptor();  // FIXME: need a way to denote optional.empty. I could use smart pointers for all
                                                // elements, but I don't love that idea. I could have flags for optional elements.

    return FullyConnected(inputDescriptor, outputWidth, weightsInitializer, useBias, biasInitializer, activation);
}*/
