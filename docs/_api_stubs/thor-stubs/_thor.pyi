"""Thor Python bindings"""

from . import (
    activations as activations,
    initializers as initializers,
    layers as layers,
    losses as losses,
    metrics as metrics,
    optimizers as optimizers,
    physical as physical,
    random as random,
    training as training
)
from thor import (
    DataType as DataType,
    MetricAggregation as MetricAggregation,
    Network as Network,
    RaggedTensor as RaggedTensor,
    Tensor as Tensor
)
from thor.constraints import (
    Max as MaxParameterConstraint,
    Min as MinParameterConstraint,
    MinMax as MinMaxParameterConstraint,
    NonNegative as NonNegativeParameterConstraint,
    NonPositive as NonPositiveParameterConstraint,
    ParameterConstraint as ParameterConstraint
)
from thor.parameters import (
    BoundParameter as BoundParameter,
    ParameterReference as ParameterReference,
    ParameterSpecification as ParameterSpecification
)
from thor.runtime import (
    PlacedNetwork as PlacedNetwork,
    StatusCode as StatusCode
)


def version() -> str: ...

def git_version() -> str: ...

__git_version__: str = '8cdad4a0-dirty'

BATCH_VALIDITY_MASK_NAME: str = '__thor_batch_validity_mask'

METRIC_AGGREGATION_NUMERATOR_NAME: str = '__thor_metric_aggregation_numerator'

METRIC_AGGREGATION_DENOMINATOR_NAME: str = '__thor_metric_aggregation_denominator'
