#include "Layer.h"
#include "NeuralNetwork/DropOut.h"

mutex DropOut::mtx;
uint64_t DropOut::seed = 0ul;
