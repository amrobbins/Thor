#include "DeepLearning/Api/Layers/Activations/Activation.h"

using namespace Thor;
using namespace std;

unordered_map<string, function<void(const nlohmann::json&, Network*)>> Activation::registry;
