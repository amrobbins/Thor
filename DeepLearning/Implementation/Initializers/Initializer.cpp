#include "DeepLearning/Implementation/Initializers/Initializer.h"

#include "DeepLearning/Implementation/ThorError.h"
using namespace std;

namespace ThorImplementation {

shared_ptr<Initializer> Initializer::clone() { THOR_UNREACHABLE(); }

}  // namespace ThorImplementation
