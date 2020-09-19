#include "Layer.h"
#include "NeuralNetwork/DropOut.h"

using namespace ThorImplementation;

mutex DropOut::mtx;
uint64_t DropOut::seed = 0ul;
