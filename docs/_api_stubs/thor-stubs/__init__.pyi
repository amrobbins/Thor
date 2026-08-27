"""Public Thor Python API."""

from collections.abc import Sequence
import enum
from typing import overload

from . import (
    activations as activations,
    constraints as constraints,
    data as data,
    ensembles as ensembles,
    initializers as initializers,
    layers as layers,
    losses as losses,
    metrics as metrics,
    optimizers as optimizers,
    parameters as parameters,
    physical as physical,
    random as random,
    runtime as runtime,
    training as training
)
from .ensembles._manifest import EnsembleModel as EnsembleModel
from .runtime import StatusCode as StatusCode


BATCH_VALIDITY_MASK_NAME: str = '__thor_batch_validity_mask'

METRIC_AGGREGATION_DENOMINATOR_NAME: str = '__thor_metric_aggregation_denominator'

METRIC_AGGREGATION_NUMERATOR_NAME: str = '__thor_metric_aggregation_numerator'

class DataType(enum.Enum):
    bool = 21

    int8 = 13

    uint8 = 17

    int16 = 14

    uint16 = 18

    int32 = 15

    uint32 = 19

    int64 = 16

    uint64 = 20

    fp16 = 10

    fp32 = 11

    fp64 = 12

    bf16 = 24

    tf32 = 25

    fp8_e4m3 = 22

    fp8_e5m2 = 23

class MetricAggregation(enum.Enum):
    MEAN_BY_EXAMPLE = 0

    SUM = 1

    MIN = 2

    MAX = 3

    RATIO = 4

class Network:
    def __init__(self, name: str) -> None: ...

    def get_network_name(self) -> str: ...

    def get_num_trainable_layers(self) -> int: ...

    def status_code_to_string(self, status_code: runtime.StatusCode) -> str: ...

    def get_last_graph_validation_error(self) -> str:
        """
        Return the detailed diagnostic from the most recent failed graph validation.

        Returns an empty string when no validation failure has been recorded.
        """

    def get_architecture_json(self) -> str: ...

    def save(self, directory: str, overwrite: bool = False) -> None: ...

    def cuda_kernel_source_info(self) -> list: ...

    def cuda_kernel_sources(self) -> list[str]: ...

    def cuda_kernel_source_info_json(self) -> str: ...

    def has_cuda_kernel_expressions(self) -> bool: ...

    def capture_cuda_kernel_save_keys_to_file(self, path: str, overwrite: bool = False) -> None:
        """
        Configure a required out-of-band key capture file for models containing
        CudaKernelExpression CUDA source. Training placement refuses to proceed for
        such networks until this is configured. The file is created immediately with a
        pending marker and overwritten with the final save-time keys when save() runs.
        """

    def clear_cuda_kernel_save_key_capture(self) -> None: ...

    def cuda_kernel_save_key_capture_configured(self) -> bool: ...

    def cuda_kernel_signing_public_keys(self) -> list[str]: ...

    def cuda_kernel_out_of_band_keys(self) -> list: ...

    def get_default_optimizer(self) -> optimizers.Optimizer: ...

    def freeze_training(self) -> None: ...

    def unfreeze_training(self) -> None: ...

    def reset_optimizers(self) -> None:
        """
        Request fresh optimizer state for the next successfully completed training phase.

        Learned parameter values are preserved. Adam moments, momentum/accumulator tensors,
        and optimizer runtime counters are initialized as they are for a newly placed network.
        If the training attempt fails and is retried, the retry also starts with fresh optimizer
        state. The request is consumed only after a training phase completes successfully.
        """

    def set_training_dropout_enabled(self, enabled: bool) -> None:
        """
        Set transient training-time dropout policy for every controllable layer in this API network.

        This does not change configured dropout probabilities and is not serialized. Validation and
        inference remain deterministic. The setting is copied into subsequently placed physical layers.
        """

    def is_training_dropout_enabled(self) -> bool: ...

    def get_num_training_dropout_controllable_layers(self) -> int: ...

    def get_trainable_parameter_references(self, training_enabled_only: bool = True) -> list[parameters.ParameterReference]: ...

    def place(self, batch_size: int, inference_only: bool = False, forced_devices: Sequence[int] = [], forced_num_stamps_per_gpu: int = 0, network_outputs_on_gpu: bool = False) -> runtime.PlacedNetwork:
        """
        Place / compile the network for execution.

        Parameters
        ----------
        batch_size : int
        inference_only : bool, default False
        forced_devices : list[int], default []
            Device ids to force placement onto. Use Network.CPU for CPU.
        forced_num_stamps_per_gpu : int, default 0
        network_outputs_on_gpu : bool, default False
            Stamp NetworkOutput layers to GPU instead of CPU. When the producer tensor is
            already on that GPU, NetworkOutput aliases the producer instead of copying, so
            ensemble runtime can aggregate member outputs on device before one final
            materialization copy.

        Returns
        -------
        thor.Network.StatusCode
        """

    load: _NetworkLoadDescriptor = ...

class RaggedTensor:
    @overload
    def __init__(self, values: Tensor, offsets: Tensor) -> None:
        """
        Construct a logical rank-1 ragged tensor from packed values and canonical row-partition offsets.

        ``values`` has shape ``[max_total_values, *trailing_dimensions]`` and ``offsets``
        has shape ``[batch_size + 1]`` with dtype ``uint32`` or ``uint64``. Offsets are
        structural metadata; they are not differentiable model values.
        """

    @overload
    def __init__(self, values: Tensor, offsets: Tensor, max_values_per_row: int) -> None:
        """
        Construct a logical rank-1 ragged tensor from packed values and offsets with an explicit maximum logical row length.
        """

    @overload
    def __init__(self, values_data_type: DataType, trailing_dimensions: Sequence[int], batch_size: int, max_total_values: int, offsets_data_type: DataType = DataType.uint32) -> None:
        """
        Construct a logical ragged tensor descriptor using Thor's canonical packed representation.
        """

    @overload
    def __init__(self, values_data_type: DataType, trailing_dimensions: Sequence[int], batch_size: int, max_total_values: int, max_values_per_row: int, offsets_data_type: DataType = DataType.uint32) -> None:
        """
        Construct a logical ragged tensor descriptor with an explicit maximum logical row length.
        """

    @property
    def values(self) -> Tensor: ...

    @property
    def offsets(self) -> Tensor: ...

    @property
    def values_data_type(self) -> DataType: ...

    @property
    def offsets_data_type(self) -> DataType: ...

    @property
    def trailing_dimensions(self) -> list[int]: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def max_total_values(self) -> int: ...

    @property
    def max_values_per_row(self) -> object: ...

    @property
    def ragged_rank(self) -> int: ...

    def get_values(self) -> Tensor: ...

    def get_offsets(self) -> Tensor: ...

    def get_values_data_type(self) -> DataType: ...

    def get_offsets_data_type(self) -> DataType: ...

    def get_trailing_dimensions(self) -> list[int]: ...

    def get_batch_size(self) -> int: ...

    def get_max_total_values(self) -> int: ...

    def get_ragged_rank(self) -> int: ...

    def get_id(self) -> int: ...

    def __eq__(self, other: RaggedTensor) -> bool: ...

    def __ne__(self, other: RaggedTensor) -> bool: ...

    def version(self) -> "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >": ...

class Tensor:
    def __init__(self, dimensions: list[int], data_type: DataType ) -> None:
        """
        A Tensor that is used to describe the shape of data and to record the
        connections between API elements.

        This tensor does not directly represent a single piece of allocated
        memory - it is possible that multiple instances of physical tensors will
        exist that represent this tensor.

        The actual allocated memory belongs to a physical tensor that is part
        of a stamped network; a corresponding physical tensor can be looked up in
        a stamped network using the ID of this API tensor.

        Parameters
        ----------
        dimensions : list[int]
            The dimensions of the tensor.
            The batch size dimension is **NOT** included here; the batch dimension
            will be created upon realization of a network via the stamping process.
        data_type : thor.DataType
            Data type of all elements in the tensor.
        """

    def get_id(self) -> int: ...

    def __eq__(self, other: Tensor) -> bool: ...

    def __ne__(self, other: Tensor) -> bool: ...

    def get_dimensions(self) -> list[int]: ...

    def get_data_type(self) -> DataType: ...

    def get_total_num_elements(self) -> int: ...

    @staticmethod
    def bytes_per_element(data_type: DataType) -> float: ...

    def get_bytes_per_element(self) -> float: ...

    def get_total_size_in_bytes(self) -> int: ...

    def version(self) -> str: ...

__git_version__: str = '7f8e9ae0-dirty'

def einsum(equation: str, *operands: Tensor, network: Network | None = None) -> Tensor:
    """
    Create a symbolic einsum operation and return its output tensor.

        The equation describes feature dimensions only; Thor's batch dimension is
        implicit and is preserved. The operation owns no trainable parameters and
        remains differentiable with respect to every operand on a live gradient path.

        When ``network`` is omitted, Thor infers the unique live Python-created
        network that contains every operand. Pass ``network=...`` explicitly when
        working with ambiguous loaded/cloned networks or tensors that are not yet
        associated with a network.
    """

__all__: list = ['BATCH_VALIDITY_MASK_NAME', 'METRIC_AGGREGATION_DENOMINATOR_NAME', 'METRIC_AGGREGATION_NUMERATOR_NAME', 'DataType', 'MetricAggregation', 'EnsembleModel', 'Network', 'RaggedTensor', 'Tensor', '__git_version__', '__version__', 'activations', 'constraints', 'data', 'ensembles', 'einsum', 'initializers', 'layers', 'losses', 'metrics', 'optimizers', 'parameters', 'physical', 'random', 'runtime', 'training']

def __dir__() -> list[str]: ...
