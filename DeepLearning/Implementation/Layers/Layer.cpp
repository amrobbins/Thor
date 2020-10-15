#include "Layer.h"
#include "NeuralNetwork/DropOut.h"

using namespace ThorImplementation;

atomic<uint64_t> Layer::nextId(2);

mutex DropOut::mtx;
uint64_t DropOut::seed = 0ul;
