import thor.optimizers
from thor import Network
from thor.layers import NetworkInput
from thor.layers import NetworkOutput
from thor.layers import Convolution2d
from thor.layers import FullyConnected
from thor import DataType
from thor import Tensor
from thor.optimizers import Adam
from thor.losses import CategoricalCrossEntropy
from thor import activations


def test_smoke():
    print(f'\nthor version: {thor.__version__}')
    print(f'thor git version: {thor.__git_version__}')

    n = Network("smoke_network")
    ni = NetworkInput(n, "smoke_input", [3, 100, 100], data_type=DataType.fp16)
    conv = Convolution2d(
        n,
        ni.get_feature_output(),
        1,
        10,
        10,
        10,
        1,
    )
    fc = FullyConnected(n, conv.get_feature_output(), 10, True, activation=activations.Relu())
    no = NetworkOutput(
        n,
        'smoke_output',
        fc.get_feature_output(),
        DataType.fp32,
    )
    t = Tensor([2, 50], data_type=DataType.uint8)
    sgd = thor.optimizers.Sgd(n)
    adam = Adam(n, epsilon=0.01)
    elu = thor.activations.Elu(alpha=0.5)
    thor.losses.CategoricalCrossEntropy
    # Would need network inputs of the right shape:
    # thor.losses.BinaryCrossEntropy(n, fc.get_feature_output(), fc.get_feature_output())
