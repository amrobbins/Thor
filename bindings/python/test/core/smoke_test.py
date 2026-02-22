import thor.optimizers
from thor import Network
from thor.layers import NetworkInput
from thor.layers import NetworkOutput
from thor.layers import Convolution2d
from thor.layers import FullyConnected
from thor import DataType
from thor import Tensor
from thor.optimizers import Adam


def test_smoke():
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
    fc = FullyConnected(
        n,
        conv.get_feature_output(),
        10,
        True,
    )
    no = NetworkOutput(
        n,
        'smoke_output',
        fc.get_feature_output(),
        DataType.fp32,
    )
    t = Tensor([2, 50], data_type=DataType.uint8)
    sgd = thor.optimizers.Sgd(n)
    adam = Adam(n)
