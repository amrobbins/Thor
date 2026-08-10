from collections.abc import Callable, Mapping, Sequence
import enum
from typing import overload

from . import numpy_dtypes as numpy_dtypes
import thor
import thor._thor.physical
from thor._thor.physical import (
    ScanOp as ScanOp,
    TensorScalarBinding as TensorScalarBinding,
    cuda_kernel_out_of_band_keys_from_json as cuda_kernel_out_of_band_keys_from_json,
    cuda_kernel_signing_public_keys_from_json as cuda_kernel_signing_public_keys_from_json,
    cuda_kernel_source_info_from_json as cuda_kernel_source_info_from_json,
    cudnn_frontend_attention_available as cudnn_frontend_attention_available
)


class DeviceType(enum.Enum):
    invalid = 0

    cpu = 1

    gpu = 2

class Placement:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, device_type: DeviceType, device_num: int = 0) -> None: ...

    def get_device_type(self) -> DeviceType: ...

    def get_device_num(self) -> int: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __eq__(self, arg: Placement, /) -> bool: ...

    def __ne__(self, arg: Placement, /) -> bool: ...

class PhysicalTensor:
    def __init__(self, placement: Placement, descriptor: PhysicalTensor.Descriptor, alignment_bytes: int = 256) -> None:
        """
        Create a PhysicalTensor with owned storage.

        Parameters
        ----------
        placement : thor.physical.TensorPlacement
        descriptor : thor.physical.TensorDescriptor
        alignment_bytes : int, default 256
            Byte alignment for pinned memory allocated on the CPU. 256 byte alignment is supported by cuda natively.
        """

    def __copy__(self) -> PhysicalTensor: ...

    def __deepcopy__(self, memo: object) -> PhysicalTensor: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def get_descriptor(self) -> PhysicalTensor.Descriptor: ...

    def get_placement(self) -> Placement: ...

    def get_dimensions(self) -> list[int]: ...

    def get_data_type(self) -> thor.DataType: ...

    def get_size_in_bytes(self) -> int: ...

    @property
    def dimensions(self) -> list[int]: ...

    @property
    def dtype(self) -> thor.DataType: ...

    def numpy(self) -> object: ...

    def copy_from_async(self, source: PhysicalTensor, stream: Stream) -> None:
        """
        Asynchronously copy tensor contents from source into this tensor using the provided stream.

        This Python binding preserves Python convenience semantics for cross-placement copies where the destination dtype is
        narrower than the source dtype by allocating an internal temporary. The C++ PhysicalTensor::copyFromAsync contract
        intentionally rejects that preserving slow path unless the caller makes the temporary/conversion explicit.

        Parameters
        ----------
        source : thor.physical.PhysicalTensor
            Source tensor to copy from.
        stream : thor.physical.Stream
            Stream used for the copy.
        """

    class Descriptor:
        def __init__(self, data_type: thor.DataType, dimensions: Sequence[int]) -> None:
            """
            TensorDescriptor(data_type, dimensions)

            Parameters
            ----------
            data_type : thor.DataType
            dimensions : list[int]
            """

        def get_data_type(self) -> thor.DataType: ...

        def get_dimensions(self) -> list[int]: ...

        def get_num_dimensions(self) -> int: ...

        def get_total_num_elements(self) -> int: ...

        def __str__(self) -> str: ...

        def __repr__(self) -> str: ...

        def __eq__(self, arg: PhysicalTensor.Descriptor, /) -> bool: ...

        def __ne__(self, arg: PhysicalTensor.Descriptor, /) -> bool: ...

        @staticmethod
        def array_size_in_bytes(num_elements: int, data_type: thor.DataType) -> int:
            """
            Return the number of bytes required to store num_elements of data_type.
            """

        @staticmethod
        def element_size_in_bytes(data_type: thor.DataType) -> float: ...

        def get_array_size_in_bytes(self) -> int: ...

        def get_element_type_name(self) -> str: ...

        @staticmethod
        def element_type_name(data_type: thor.DataType) -> str: ...

        def is_integral_type(self) -> bool: ...

        @staticmethod
        def is_integral_data_type(data_type: thor.DataType) -> bool: ...

        def is_boolean_type(self) -> bool: ...

        @staticmethod
        def is_boolean_data_type(data_type: thor.DataType) -> bool: ...

        def is_signed_type(self) -> bool: ...

        @staticmethod
        def is_signed_data_type(data_type: thor.DataType) -> bool: ...

        def reshape(self, new_dimensions: Sequence[int]) -> None: ...

        def flat_index(self, element: Sequence[int]) -> int:
            """
            Return the flat index corresponding to a multidimensional element index.

            Parameters
            ----------
            element : Sequence[int]
                One index per tensor dimension. Its length must match the number
                of dimensions, and each index must be within bounds.

            Returns
            -------
            int
                The flattened linear index of the element in row-major order.
            """

        def dimensional_index(self, flat_index: int) -> list[int]:
            """
            Return the per dimension indexes of an element, given its flat index (element offset from the beginning of the tensor).

            Parameters
            ----------
            flat_index : int
                Offset of the element from the beginning of the tensor.

            Returns
            -------
            Sequence[int]
                One index per tensor dimension, that addresses the element at offset flat_index.
            """

        def dimension_stride(self, axis: int) -> int:
            """
            Return the number of elements contained at the specified axis, before the next index in the axis.
            For example:

                if tensor has shape [2][3][4]
                tensor.dimension_stride(axis=0) == 12
                tensor.dimension_stride(axis=1) == 4
                tensor.dimension_stride(axis=2) == 1

            Parameters
            ----------
            axis : int
                The dimension for which the stride is computed.

            Returns
            -------
            int
                The number of elements between subsequent indexes in the specified dimension.
            """

    @overload
    def clone(self) -> PhysicalTensor:
        """
        Create another tensor like this one with the same placement, data type, and dimensions.
        """

    @overload
    def clone(self, new_placement: Placement) -> PhysicalTensor:
        """
        Create another tensor like this one but with a different placement.

        Parameters
        ----------
        new_placement : thor.physical.Placement
            Destination placement for the cloned tensor.
        """

    @overload
    def clone(self, new_data_type: thor.DataType) -> PhysicalTensor:
        """
        Create another tensor like this one but with a different data type.

        Parameters
        ----------
        new_data_type : thor.DataType
            Destination data type for the cloned tensor.
        """

    @overload
    def clone(self, new_placement: Placement, new_data_type: thor.DataType) -> PhysicalTensor:
        """
        Create another tensor like this one but with a different placement and data type.

        Parameters
        ----------
        new_placement : thor.physical.Placement
            Destination placement for the cloned tensor.
        new_data_type : thor.DataType
            Destination data type for the cloned tensor.
        """

    @overload
    def clone(self, new_dimensions: Sequence[int]) -> PhysicalTensor:
        """
        Create another tensor like this one but with a different dimensions.

        Parameters
        ----------
        new_dimensions : list[int]
            New tensor dimensions.
        """

    @overload
    def clone_copy_async(self, new_placement: Placement, stream: Stream) -> PhysicalTensor:
        """
        Create a clone of this tensor with a new placement, then copy this tensor into it asynchronously on the given stream.

        Parameters
        ----------
        new_placement : thor.physical.Placement
            Destination placement for the cloned tensor.
        stream : thor.physical.Stream
            Stream used for the asynchronous copy.

        Returns
        -------
        thor.physical.PhysicalTensor
            The cloned tensor. The copy operation has been scheduled on the stream.
        """

    @overload
    def clone_copy_async(self, new_data_type: thor.DataType, stream: Stream) -> PhysicalTensor:
        """
        Create a clone of this tensor with a new data type, then copy this tensor into it asynchronously on the given stream.

        Parameters
        ----------
        new_data_type : thor.DataType
            Destination data type for the cloned tensor.
        stream : thor.physical.Stream
            Stream used for the asynchronous copy.

        Returns
        -------
        thor.physical.PhysicalTensor
            The cloned tensor. The copy operation has been scheduled on the stream.
        """

class PhysicalRaggedTensor:
    def __init__(self, values: PhysicalTensor, offsets: PhysicalTensor) -> None: ...

    @property
    def values(self) -> PhysicalTensor: ...

    @property
    def offsets(self) -> PhysicalTensor: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def max_total_values(self) -> int: ...

    @property
    def values_data_type(self) -> thor.DataType: ...

    @property
    def offsets_data_type(self) -> thor.DataType: ...

    def get_values(self) -> PhysicalTensor: ...

    def get_offsets(self) -> PhysicalTensor: ...

class Event:
    def __init__(self, gpu_num: int = 0, enable_timing: bool = False, expecting_host_to_wait: bool = False) -> None:
        """
        Event(gpu_num=0, enable_timing=False, expecting_host_to_wait=False)

        Create a CUDA event.

        Parameters
        ----------
        gpu_num : int, default 0
        enable_timing : bool, default False
        expecting_host_to_wait : bool, default False
        """

    def __copy__(self) -> Event: ...

    def __deepcopy__(self, memo: object) -> Event: ...

    def get_gpu_num(self) -> int: ...

    def get_id(self) -> int: ...

    def record(self, stream: Stream) -> None:
        """Record this event on the given stream."""

    def synchronize(self) -> None:
        """Block until this event is completed."""

    def synchronize_and_report_elapsed_time_ms(self, start_event: Event) -> float:
        """
        Synchronize this event and return elapsed time in milliseconds since start_event.

        Returns
        -------
        float
        """

    def __repr__(self) -> str: ...

class Stream:
    @overload
    def __init__(self, gpu_num: int = 0) -> None:
        """
        Stream(gpu_num=0)

        Create a CUDA stream on the specified GPU.
        Priority is always REGULAR in the Python API.

        Parameters
        ----------
        gpu_num : int, default 0
        """

    @overload
    def __init__(self, placement: Placement) -> None:
        """
        Stream(placement)

        Create a CUDA stream based on a tensor placement.
        Priority is always REGULAR in the Python API.

        Parameters
        ----------
        placement : thor.physical.Placement
        """

    def __copy__(self) -> Stream: ...

    def __deepcopy__(self, memo: object) -> Stream: ...

    def get_gpu_num(self) -> int: ...

    def get_id(self) -> int: ...

    def synchronize(self) -> None:
        """Block until all queued work on this stream has completed."""

    @staticmethod
    def device_synchronize(gpu_num: int = 0) -> None:
        """
        device_synchronize(gpu_num=0)

        Block until all work on the specified device has completed.
        """

    def put_event(self, enable_timing: bool = False, expecting_host_to_wait: bool = False) -> Event:
        """
        Create and record an event on this stream.

        Returns
        -------
        thor.physical.Event
        """

    def wait_event(self, event: Event) -> None:
        """Make this stream wait until the given event is completed."""

    def __repr__(self) -> str: ...

class MachineEvaluator:
    @staticmethod
    def instance() -> MachineEvaluator:
        """Return the singleton MachineEvaluator instance."""

    def get_current_gpu_num(self) -> int: ...

    def get_connection_speed_rankings(self, source_gpu_num: int) -> list[MachineEvaluator.GpuConnectionRanking]: ...

    def is_peer_to_peer_available(self, source_gpu_num: int, dest_gpu_num: int) -> bool: ...

    @overload
    def get_gpu_type(self, gpu_num: int) -> str: ...

    @overload
    def get_gpu_type(self) -> str:
        """Return the GPU type for the current GPU."""

    def get_gpu_pci_bus_id(self, gpu_num: int) -> int: ...

    def get_gpu_num_from_bus_id(self, gpu_bus_id: int) -> int: ...

    def get_adjacent_higher_gpu(self, gpu_num: int) -> int: ...

    def get_adjacent_lower_gpu(self, gpu_num: int) -> int: ...

    def get_ordered_gpus(self) -> list[int]: ...

    def get_num_gpus(self) -> int: ...

    @overload
    def get_num_multi_processors(self, gpu_num: int) -> int: ...

    @overload
    def get_num_multi_processors(self) -> int:
        """Return SM count for the current GPU."""

    def get_total_global_mem_bytes(self, gpu_num: int) -> int: ...

    def get_free_mem_bytes(self, gpu_num: int) -> int: ...

    @staticmethod
    def swap_active_device(new_gpu_num: int) -> int: ...

    class GpuConnectionRanking:
        def __init__(self) -> None: ...

        @property
        def peer_gpu_num(self) -> int: ...

        @peer_gpu_num.setter
        def peer_gpu_num(self, arg: int, /) -> None: ...

        @property
        def is_peer_to_peer_supported(self) -> bool: ...

        @is_peer_to_peer_supported.setter
        def is_peer_to_peer_supported(self, arg: bool, /) -> None: ...

        @property
        def peer_to_peer_speed_ranking(self) -> int: ...

        @peer_to_peer_speed_ranking.setter
        def peer_to_peer_speed_ranking(self, arg: int, /) -> None: ...

        def __lt__(self, arg: MachineEvaluator.GpuConnectionRanking, /) -> bool: ...

        def __repr__(self) -> str: ...

class ScopedGpu:
    def __init__(self, gpu_num: int) -> None: ...

    def __enter__(self) -> ScopedGpu: ...

    def __exit__(self, exc_type: object | None = None, exc: object | None = None, tb: object | None = None) -> None: ...

class AttentionTensorLayout(enum.Enum):
    """
    Attention tensor layout used by cuDNN SDPA expression stages.

    The semantic tensor shape is always ``[B, H, S, D]``.  ``bhsd`` stores that order
    directly.  ``bshd`` stores batch, sequence, heads, head dimension.  Ragged/packed
    THD attention requires BSHD physical layouts for Q/K/V/O so ragged offsets index
    packed token-contiguous storage.
    """

    bhsd = 0

    bshd = 1

class AttentionMaskKind(enum.Enum):
    """
    Mask kinds supported by Thor's cuDNN SDPA path.

    ``causal_top_left`` and ``sliding_window_top_left`` use standard top-left diagonal
    semantics.  ``causal_bottom_right`` and ``sliding_window_bottom_right`` support
    decode-style alignment, but production cuDNN primary SDPA currently requires
    additive bias, ALiBi, and dropout to be disabled for bottom-right/decode masks.
    ALiBi requires a causal/sliding diagonal mask with ``diagonal_right_bound == 0``.
    """

    none = 0

    causal_top_left = 1

    causal_bottom_right = 2

    sliding_window_top_left = 3

    sliding_window_bottom_right = 4

class RotaryScalingKind(enum.Enum):
    """
    RoPE scaling parameterization for ``Expression.rotary_position_embedding`` and
    ``thor.layers.Attention``.  The high-level Attention layer supports ``none``,
    ``linear``, ``dynamic_ntk``, ``yarn``, ``longrope``, and ``llama3``.
    """

    none = 0

    linear = 1

    dynamic_ntk = 2

    yarn = 3

    longrope = 4

    llama3 = 5

class CudaKernelDimExpr:
    @staticmethod
    def constant(value: int) -> CudaKernelDimExpr: ...

    @staticmethod
    def dim(tensor_name: str, axis: int) -> CudaKernelDimExpr: ...

    @staticmethod
    def numel(tensor_name: str) -> CudaKernelDimExpr: ...

    def describe(self) -> str: ...

    def __repr__(self) -> str: ...

class CudaKernelLaunchConfig:
    def __init__(self, grid: object, block: object, dynamic_shared_bytes: int = 0) -> None:
        """
        CUDA launch configuration for a CudaKernelExpression.

        ``grid`` and ``block`` may be 1-, 2-, or 3-element integer sequences. Missing
        trailing dimensions default to 1. ``dynamic_shared_bytes`` is passed as the
        kernel launch dynamic shared-memory byte count.
        """

    @staticmethod
    def grid_1d(elements: int, block_size: int = 256, dynamic_shared_bytes: int = 0) -> CudaKernelLaunchConfig: ...

    @property
    def dynamic_shared_bytes(self) -> int: ...

    @dynamic_shared_bytes.setter
    def dynamic_shared_bytes(self, arg: int, /) -> None: ...

    @property
    def grid(self) -> list[int]: ...

    @property
    def block(self) -> list[int]: ...

class CudaKernelLaunchContext:
    def dim(self, tensor_name: str, axis: int) -> int: ...

    def numel(self, tensor_name: str) -> int: ...

    def dtype(self, tensor_name: str) -> thor.DataType: ...

    @property
    def device_num(self) -> int: ...

class CudaKernelExpression:
    @staticmethod
    def builder(name: str) -> CudaKernelExpressionBuilder: ...

    @staticmethod
    def dim(tensor_name: str, axis: int) -> CudaKernelDimExpr: ...

    @staticmethod
    def numel(tensor_name: str) -> CudaKernelDimExpr: ...

    @staticmethod
    def constant_dim(value: int) -> CudaKernelDimExpr: ...

    def name(self) -> str: ...

    def input_names(self) -> list[str]: ...

    def tensor_input_names(self) -> list[str]: ...

    def output_names(self) -> list[str]: ...

    @property
    def source(self) -> str: ...

    @property
    def compiled_source(self) -> str: ...

    @property
    def loaded_source_compilation_allowed(self) -> bool: ...

    def source_info(self) -> dict: ...

    def source_info_json(self) -> str: ...

    def apply(self, inputs: Mapping[str, Expression]) -> Outputs: ...

    def __call__(self, inputs: Mapping[str, Expression]) -> Outputs: ...

    def as_dynamic_expression(self) -> DynamicExpression: ...

    def stamp(self, inputs: Mapping[str, PhysicalTensor], preallocated_outputs: Mapping[str, PhysicalTensor], stream: Stream, tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}) -> Equation:
        """
        Compile, bind, and stamp this CUDA kernel expression directly.

        Most users should use ``apply(...)`` to stitch the custom kernel into a normal
        Thor expression graph. This direct path is useful for low-level tests and
        standalone custom kernels.
        """

class CudaKernelExpressionBuilder:
    def __init__(self, name: str) -> None: ...

    def source(self, cuda_source: str) -> CudaKernelExpressionBuilder: ...

    def entry(self, entrypoint: str) -> CudaKernelExpressionBuilder: ...

    def input(self, name: str, dtype: thor.DataType) -> CudaKernelExpressionBuilder: ...

    def tensor_runtime_scalar_input(self, name: str, dtype: thor.DataType) -> CudaKernelExpressionBuilder: ...

    def host_runtime_scalar_input(self, name: str, dtype: thor.DataType) -> CudaKernelExpressionBuilder: ...

    def output(self, name: str, dtype: thor.DataType, shape: Sequence) -> CudaKernelExpressionBuilder: ...

    def output_like(self, name: str, dtype: thor.DataType, input_name: str) -> CudaKernelExpressionBuilder: ...

    def scalar(self, name: str, type: thor.DataType, value: object) -> CudaKernelExpressionBuilder: ...

    def launch(self, launch: Callable) -> CudaKernelExpressionBuilder: ...

    def launch_grid_1d(self, elements: object, block_size: int = 256, dynamic_shared_bytes: int = 0) -> CudaKernelExpressionBuilder: ...

    def backward(self, forward_output_name: str, backward_kernel: CudaKernelExpression, upstream_gradient_input_name: str, input_gradients: Mapping[str, str]) -> CudaKernelExpressionBuilder:
        """
        Attach an explicit vector-Jacobian-product CUDA kernel for one forward output.

        The backward kernel receives the upstream gradient through
        ``upstream_gradient_input_name``. Any other backward-kernel inputs must have
        the same names, dtypes, and kinds as forward-kernel inputs; Thor binds those
        forward values automatically. ``input_gradients`` maps backward-kernel output
        names to the corresponding forward-kernel input whose gradient they produce.
        Forward inputs omitted from the mapping receive no gradient contribution.
        """

    def build(self) -> CudaKernelExpression: ...

sum: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.sum

min: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.min

max: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.max

product: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.product

arg_min: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.arg_min

arg_max: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.arg_max

COPY_DIM: int = 0

INFER_DIM: int = 18446744073709551615

class Expression:
    def __init__(self, arg: float, /) -> None: ...

    @staticmethod
    def input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression:
        """
        Create an input expression.

        Parameters
        ----------
        name : str
            Input name.
        output_dtype : thor.DataType | None
            Optional dtype to cast the input value to when it enters the expression graph.
            The actual bound runtime tensor may have a different dtype.

        Returns
        -------
        thor.physical.Expression
            An Expression representing that input.
        """

    @staticmethod
    def runtime_scalar(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression:
        """
        Create a runtime-bound scalar input expression.

        Parameters
        ----------
        name : str
            Runtime scalar input name.
        output_dtype : thor.DataType | None
            Optional dtype cast applied to the runtime scalar as it enters the graph.
            Currently runtime scalar bindings are passed as fp32 values.

        Returns
        -------
        thor.physical.Expression
            An Expression representing that runtime scalar input.
        """

    @staticmethod
    def tensor_runtime_scalar(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression:
        """
        Create a GPU tensor-backed runtime scalar input expression.

        The scalar is loaded from a bound GPU buffer at stamp time using a
        TensorScalarBinding (buffer, byte_offset, source_dtype).
        """

    def with_dtypes(self, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression:
        """
        Return a new expression whose result node has local dtype overrides.

        This only annotates the current expression result node. It does not recursively
        rewrite the dtypes of ancestor nodes in the subexpression.

        Parameters
        ----------
        output_dtype : thor.DataType | None
            Optional output dtype override for this expression node.
        compute_dtype : thor.DataType | None
            Optional compute dtype override for this expression node.

        Returns
        -------
        thor.physical.Expression
            A new Expression with the requested local dtype overrides applied to its
            result node.
        """

    def with_output_dtype(self, output_dtype: thor.DataType) -> Expression:
        """
        Return a new expression whose result node uses the requested output dtype.
        """

    def with_compute_dtype(self, compute_dtype: thor.DataType) -> Expression:
        """
        Return a new expression whose result node uses the requested compute dtype.
        """

    def cast(self, dtype: thor.DataType) -> Expression:
        """
        Return a value-cast expression with the requested output dtype.

        Unlike ``with_output_dtype()``, this inserts an explicit cast node, so boolean
        masks can be converted to integral tensors before integer-only stages such as
        prefix-sum scan.
        """

    def reshape(self, shape: Sequence[int]) -> Expression:
        """
        Return a metadata-only reshape expression.

        For contiguous tensors this is planned as a value alias rather than a materializing
        fused kernel when no dtype conversion is requested.
        """

    def strided_view(self, shape: Sequence[int], strides: Sequence[int], element_offset: int = 0) -> Expression:
        """
        Return a zero-materialization storage alias with explicit element strides.

        The alias shares the source allocation, starts at element_offset elements from the
        source tensor's visible base pointer, and indexes the requested shape using the
        provided element strides. This is intended for layout/descriptor adapters such as
        packed-QKV attention views; generic fused kernels should materialize or lower a
        layout-aware kernel before consuming non-dense views.
        """

    def take_along_axis(self, indices: Expression, axis: int = -1) -> Expression:
        """
        Gather values from this expression along one axis using integer indices.

        The input and indices tensors must have the same rank. Dimensions must match on
        all axes except the gathered axis, and the output shape is the indices shape.
        Indices must be UINT32 or UINT64. The default axis=-1 gathers along the final
        axis.
        """

    def scan(self, op: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.sum, axis: int = -1, inclusive: bool = True) -> Expression:
        """
        Prefix scan along the final contiguous axis.

        Supported value ops are ``ScanOp.sum``, ``ScanOp.min``, ``ScanOp.max``, and
        ``ScanOp.product``. Index ops ``ScanOp.arg_min`` and ``ScanOp.arg_max`` return
        UINT32 flattened input indices for the winning prefix element. ``inclusive=True``
        includes the current element; ``inclusive=False`` returns each prefix before the
        current element. Value-scan output shape and dtype match the input; use
        ``cast(thor.DataType.uint32)`` to scan masks as counts.
        """

    def prefix_count(self, inclusive: bool = True, axis: int = -1) -> Expression:
        """
        Prefix count of true/nonzero mask elements using UINT32 sum scan.

        This is equivalent to ``mask.cast(thor.DataType.uint32).scan(ScanOp.sum, axis, inclusive)`` and is
        intended for nonzero/compaction/ragged-packing index generation.
        """

    def segmented_scan(self, offsets: Expression, op: thor._thor.physical.ScanOp = thor._thor.physical.ScanOp.sum, inclusive: bool = True) -> Expression:
        """
        Ragged segmented prefix scan over the flattened input using a rank-1 offsets
        tensor of shape [num_segments + 1]. Offsets must be UINT32 or UINT64. The
        output shape and dtype match the input for value scans. Arg scans return UINT32
        flattened input indices for the winning prefix element.
        """

    def scan_with_indices(self, op: thor._thor.physical.ScanOp, axis: int = -1, inclusive: bool = True) -> tuple:
        """
        Return paired prefix min/max values and UINT32 flattened input indices from one coalescible scan stage.
        """

    def segmented_scan_with_indices(self, offsets: Expression, op: thor._thor.physical.ScanOp, inclusive: bool = True) -> tuple:
        """
        Return paired ragged segmented prefix min/max values and UINT32 flattened input indices from one coalescible scan stage.
        """

    def exclusive_scan_sum(self, axis: int = -1) -> Expression:
        """Exclusive prefix sum scan along the final contiguous axis."""

    def inclusive_scan_sum(self, axis: int = -1) -> Expression:
        """Inclusive prefix sum scan along the final contiguous axis."""

    @staticmethod
    def constant_scalar(value: float) -> Expression:
        """Create a floating-point scalar constant expression."""

    def __add__(self, other: Expression) -> Expression: ...

    def __sub__(self, other: Expression) -> Expression: ...

    def __mul__(self, other: Expression) -> Expression: ...

    def __truediv__(self, other: Expression) -> Expression: ...

    def __pow__(self, arg: Expression, /) -> Expression: ...

    def __eq__(self, other: Expression) -> Expression: ...

    def __ne__(self, other: Expression) -> Expression: ...

    def __lt__(self, other: Expression) -> Expression: ...

    def __le__(self, other: Expression) -> Expression: ...

    def __gt__(self, other: Expression) -> Expression: ...

    def __ge__(self, other: Expression) -> Expression: ...

    def __and__(self, other: Expression) -> Expression: ...

    def __or__(self, other: Expression) -> Expression: ...

    def __invert__(self) -> Expression: ...

    def equal(self, other: Expression) -> Expression: ...

    def not_equal(self, other: Expression) -> Expression: ...

    def less_than(self, other: Expression) -> Expression: ...

    def less_equal(self, other: Expression) -> Expression: ...

    def greater_than(self, other: Expression) -> Expression: ...

    def greater_equal(self, other: Expression) -> Expression: ...

    def logical_and(self, other: Expression) -> Expression: ...

    def logical_or(self, other: Expression) -> Expression: ...

    def logical_not(self) -> Expression: ...

    def select(self, true_value: Expression, false_value: Expression) -> Expression: ...

    @staticmethod
    def where(condition: Expression, true_value: Expression, false_value: Expression) -> Expression:
        """
        Elementwise conditional selection.

        Returns true_value where condition is true, otherwise false_value. The
        condition must be boolean and all three operands use normal Thor broadcast
        semantics.
        """

    def __radd__(self, other: Expression) -> Expression: ...

    def __rsub__(self, other: Expression) -> Expression: ...

    def __rmul__(self, other: Expression) -> Expression: ...

    def __rtruediv__(self, other: Expression) -> Expression: ...

    @overload
    def __rpow__(self, other: Expression) -> Expression: ...

    @overload
    def __rpow__(self, other: Expression) -> Expression: ...

    def __rand__(self, other: Expression) -> Expression: ...

    def __ror__(self, other: Expression) -> Expression: ...

    def __matmul__(self, other: Expression) -> Expression: ...

    def __rmatmul__(self, other: Expression) -> Expression: ...

    def __imatmul__(self, other: Expression) -> Expression: ...

    @overload
    def __neg__(self) -> Expression: ...

    @overload
    def __neg__(self) -> Expression: ...

    def transpose(self) -> Expression:
        """
        Return an expression with the last two dimensions swapped.

        For rank-2 tensors this is a matrix transpose. For rank > 2, this is
        a batched transpose over the trailing matrix dimensions. A final transpose
        can be folded into a fused tiled materialization kernel.
        """

    @property
    def T(self) -> Expression:
        """Shorthand for ``self.transpose()``."""

    @staticmethod
    def conv2d(x: Expression, w: Expression, stride_h: int = 1, stride_w: int = 1, pad_h: int = 0, pad_w: int = 0, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression: ...

    @staticmethod
    def conv3d(x: Expression, w: Expression, stride_d: int = 1, stride_h: int = 1, stride_w: int = 1, pad_d: int = 0, pad_h: int = 0, pad_w: int = 0, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression: ...

    @staticmethod
    def matmul(a: Expression, b: Expression, transpose_a: bool = False, transpose_b: bool = False, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression: ...

    @staticmethod
    def gemm(a: Expression, b: Expression, c: Expression, alpha: Expression = 1.0, beta: Expression = 1.0, transpose_a: bool = False, transpose_b: bool = False, transpose_c: bool = False, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression: ...

    @staticmethod
    def embedding_lookup(indices: Expression, weights: Expression, padding_index: object | None = None, output_dtype: object | None = None) -> Expression:
        """
        Embedding lookup expression. The indices tensor must be uint8, uint16, uint32, or uint64. The weights tensor must have shape
        [vocabulary_size, embedding_dim], and the output shape is indices.shape + [embedding_dim]. When padding_index is set,
        matching rows are written as zeros without reading the weight table.
        """

    @staticmethod
    def rotary_position_embedding(input: Expression, sequence_axis: int = 2, head_dim_axis: int = 3, rotary_dim: int = 0, base: float = 10000.0, position_offset: int = 0, interleaved: bool = False, inverse: bool = False, scaling_kind: RotaryScalingKind = RotaryScalingKind.none, scaling_factor: float = 1.0, original_max_position_embeddings: int = 0, attention_factor: float | None = None, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0, llama3_low_freq_factor: float = 1.0, llama3_high_freq_factor: float = 4.0, long_rope_short_factors: Sequence[float] = [], long_rope_long_factors: Sequence[float] = [], output_dtype: object | None = None, compute_dtype: object | None = None, allow_in_place_materialization: bool = False) -> Expression:
        """
        Apply rotary positional embedding as a fused expression primitive.

        The tensor is interpreted as having a sequence axis and an innermost head-dim
        axis.  rotary_dim=0 rotates the full head dimension; otherwise only the leading
        rotary_dim channels are rotated.  The inverse flag applies the transpose rotation,
        which is used by autodiff.
        """

    @staticmethod
    def rope(input: Expression, sequence_axis: int = 2, head_dim_axis: int = 3, rotary_dim: int = 0, base: float = 10000.0, position_offset: int = 0, interleaved: bool = False, inverse: bool = False, scaling_kind: RotaryScalingKind = RotaryScalingKind.none, scaling_factor: float = 1.0, original_max_position_embeddings: int = 0, attention_factor: float | None = None, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0, llama3_low_freq_factor: float = 1.0, llama3_high_freq_factor: float = 4.0, long_rope_short_factors: Sequence[float] = [], long_rope_long_factors: Sequence[float] = [], output_dtype: object | None = None, compute_dtype: object | None = None, allow_in_place_materialization: bool = False) -> Expression:
        """Alias for rotary_position_embedding()."""

    @staticmethod
    def scaled_dot_product_attention(q: Expression, k: Expression, v: Expression, q_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, k_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, v_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, o_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, mask_kind: AttentionMaskKind = AttentionMaskKind.none, diagonal_left_bound: int = 0, diagonal_right_bound: int = 0, attention_scale: object | None = None, use_alibi_mask: bool = False, output_dtype: object | None = None, compute_dtype: object | None = None, bias: object | None = None, q_seq_len: object | None = None, kv_seq_len: object | None = None, q_ragged_offsets: object | None = None, kv_ragged_offsets: object | None = None, page_table_k: object | None = None, page_table_v: object | None = None, paged_kv_max_sequence_length: int = 0, dropout_probability: float = 0.0, dropout_seed: object | None = None, dropout_offset: object | None = None) -> Expression:
        """
        Create a cuDNN scaled-dot-product attention expression stage.

        All tensors use semantic shape ``[B, H, S, D]``.  Layout arguments describe how
        those tensors are handed to cuDNN.  The default layout is BHSD, matching Thor's
        row-major physical tensor layout for rank-4 attention inputs.  ``output_dtype``
        should normally match Q/K/V for the current cuDNN SDPA path; ``compute_dtype``
        should normally be ``thor.DataType.fp32``.

        FP16/BF16 production support:

        * Q/K/V/O must all use the same FP16 or BF16 dtype.  Forward and backward are
          supported for self-attention, cross-attention, MHA, GQA, and MQA.
        * Supported masks are ``none``, ``causal_top_left``, ``causal_bottom_right``,
          ``sliding_window_top_left``, and ``sliding_window_bottom_right``.
        * ALiBi requires a causal/sliding diagonal mask and ``diagonal_right_bound == 0``.
        * ``bias`` is additive score-space bias in ``[1|B, 1|Hq, 1|Sq, 1|Skv]`` semantic
          order and must use the compute dtype.  Forward supports sequence broadcast.
          Backward materializes sequence-broadcast bias to dense score space before
          cuDNN backward, then explicitly reduces dBias back to the requested bias shape.
        * When ``q_ragged_offsets`` and ``kv_ragged_offsets`` are provided, they must be
          canonical uint32/uint64 GPU token row partitions with shape ``[B + 1]``.  They
          are mutually exclusive with ``q_seq_len``/``kv_seq_len``; Thor derives cuDNN's
          private int32 sequence lengths and independent Q/K/V/O element offsets on-device.
          Ragged + additive-bias forward is supported, but ragged + additive-bias
          backward is rejected.
        * When ``dropout_probability > 0``, ``dropout_seed`` and ``dropout_offset`` must
          be int64 GPU scalar expressions with shape ``[1, 1, 1, 1]``.  They are passed
          to cuDNN's Philox attention dropout path.

        Paged KV cache:

        * ``page_table_k`` and ``page_table_v`` must be int32 GPU tensors with shape
          ``[B, 1, ceil(Skv / block_size), 1]``.  Paged-KV attention requires
          ``q_seq_len`` and ``kv_seq_len`` and a positive ``paged_kv_max_sequence_length``.
        * The production paged-KV path is FP16/BF16 forward-only/inference-only.  Bias,
          dropout, ragged offsets, and backward are rejected for paged KV.

        FP8 support:

        * FP8 is exposed by the lower-level FP8-specific expression path and by
          ``thor.layers.ScaledDotProductAttention`` with explicit scale/descale/amax
          tensors.  This generic expression wrapper documents the same validated surface:
          forward-only, same FP8 format for Q/K/V/O, head dimensions multiples of 16 and
          ``<= 128``, no additive bias, no dropout, no ALiBi, no ragged, no paged KV, no
          bottom-right/decode or sliding-window masks, and no decode-style ``Sq=1, Skv>1``.
        * FP8 padding masks / sequence lengths are supported for forward.

        Important combination rules:

        * Bottom-right/decode masks currently require additive bias, ALiBi, and dropout
          to be disabled in the production cuDNN primary SDPA path.
        * Experimental cuDNN support-surface probe environment variables can bypass some
          guards for measurement only; probe-only combinations are not support guarantees.
        """

    @staticmethod
    def attention(q: Expression, k: Expression, v: Expression, q_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, k_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, v_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, o_layout: AttentionTensorLayout = AttentionTensorLayout.bhsd, mask_kind: AttentionMaskKind = AttentionMaskKind.none, diagonal_left_bound: int = 0, diagonal_right_bound: int = 0, attention_scale: object | None = None, use_alibi_mask: bool = False, output_dtype: object | None = None, compute_dtype: object | None = None, bias: object | None = None, q_seq_len: object | None = None, kv_seq_len: object | None = None, q_ragged_offsets: object | None = None, kv_ragged_offsets: object | None = None, page_table_k: object | None = None, page_table_v: object | None = None, paged_kv_max_sequence_length: int = 0, dropout_probability: float = 0.0, dropout_seed: object | None = None, dropout_offset: object | None = None) -> Expression:
        """Alias for scaled_dot_product_attention()."""

    @staticmethod
    def min(a: Expression, b: Expression) -> Expression: ...

    @staticmethod
    def max(a: Expression, b: Expression) -> Expression: ...

    @overload
    @staticmethod
    def clamp(x: Expression, lower_bound: float, upper_bound: float) -> Expression:
        """
        Clamp x elementwise to the inclusive scalar range [lower_bound, upper_bound].

        This lowers to max(x, lower_bound) followed by min(..., upper_bound), so it
        remains a normal fusable expression graph.
        """

    @overload
    @staticmethod
    def clamp(x: Expression, lower_bound: Expression, upper_bound: Expression) -> Expression:
        """
        Clamp x elementwise to the inclusive expression range [lower_bound, upper_bound].

        The bounds may be scalar expressions or broadcast-compatible expressions. This
        lowers to max(x, lower_bound) followed by min(..., upper_bound), so it remains a
        normal fusable expression graph.
        """

    @staticmethod
    def dot_product(a: Expression, b: Expression, compute_dtype: object | None = None) -> Expression:
        """
        Return the dot product of two broadcast-compatible expressions.

        This lowers to elementwise multiply followed by reduce_sum over all axes, with
        all singleton dimensions squeezed so the result has shape [1].
        """

    @staticmethod
    def outer_product(a: Expression, b: Expression, output_dtype: object | None = None, compute_dtype: object | None = None) -> Expression:
        """
        Return the outer product of two rank-1 vector expressions.

        This lowers to matmul(unsqueeze(a, 1), unsqueeze(b, 0)), producing an [N, M]
        matrix for vector inputs shaped [N] and [M].
        """

    @staticmethod
    def abs(x: Expression) -> Expression:
        """Return the absolute value of the input expression x"""

    @staticmethod
    def ceil(x: Expression) -> Expression: ...

    @staticmethod
    def floor(x: Expression) -> Expression: ...

    @staticmethod
    def round(x: Expression) -> Expression: ...

    @staticmethod
    def trunc(x: Expression) -> Expression: ...

    @staticmethod
    def sin(x: Expression) -> Expression: ...

    @staticmethod
    def cos(x: Expression) -> Expression: ...

    @staticmethod
    def tan(x: Expression) -> Expression: ...

    @staticmethod
    def csc(x: Expression) -> Expression: ...

    @staticmethod
    def sec(x: Expression) -> Expression: ...

    @staticmethod
    def cot(x: Expression) -> Expression: ...

    @staticmethod
    def asin(x: Expression) -> Expression: ...

    @staticmethod
    def acos(x: Expression) -> Expression: ...

    @staticmethod
    def atan(x: Expression) -> Expression: ...

    @staticmethod
    def acsc(x: Expression) -> Expression: ...

    @staticmethod
    def asec(x: Expression) -> Expression: ...

    @staticmethod
    def acot(x: Expression) -> Expression: ...

    @staticmethod
    def sinh(x: Expression) -> Expression: ...

    @staticmethod
    def cosh(x: Expression) -> Expression: ...

    @staticmethod
    def csch(x: Expression) -> Expression: ...

    @staticmethod
    def sech(x: Expression) -> Expression: ...

    @staticmethod
    def coth(x: Expression) -> Expression: ...

    @staticmethod
    def asinh(x: Expression) -> Expression: ...

    @staticmethod
    def acosh(x: Expression) -> Expression: ...

    @staticmethod
    def atanh(x: Expression) -> Expression: ...

    @staticmethod
    def acsch(x: Expression) -> Expression: ...

    @staticmethod
    def asech(x: Expression) -> Expression: ...

    @staticmethod
    def acoth(x: Expression) -> Expression: ...

    @staticmethod
    def erf(x: Expression) -> Expression: ...

    @staticmethod
    def erfc(x: Expression) -> Expression: ...

    @staticmethod
    def erfcx(x: Expression) -> Expression: ...

    @staticmethod
    def erfinv(x: Expression) -> Expression: ...

    @staticmethod
    def erfcinv(x: Expression) -> Expression: ...

    @staticmethod
    def tgamma(x: Expression) -> Expression: ...

    @staticmethod
    def lgamma(x: Expression) -> Expression: ...

    @staticmethod
    def digamma(x: Expression) -> Expression: ...

    @staticmethod
    def exp(x: Expression) -> Expression: ...

    @staticmethod
    def expm1(x: Expression) -> Expression: ...

    @staticmethod
    def exp2(x: Expression) -> Expression: ...

    @staticmethod
    def exp10(x: Expression) -> Expression: ...

    @staticmethod
    def ln(x: Expression) -> Expression:
        """Return the elementwise natural logarithm of the input expression x"""

    @staticmethod
    def log(x: Expression, base: float = 2.718281828459045) -> Expression: ...

    @staticmethod
    def log1p(x: Expression) -> Expression: ...

    @staticmethod
    def log2(x: Expression) -> Expression: ...

    @staticmethod
    def log10(x: Expression) -> Expression: ...

    @staticmethod
    def sqrt(x: Expression) -> Expression:
        """Return the elementwise square root of the input expression x"""

    @staticmethod
    def tanh(x: Expression) -> Expression:
        """
        Return the elementwise hyperbolic tangent of the input expression x.

        This lowers to Thor's TANH expression op, which is emitted with CUDA's built-in tanh implementation.
        """

    @staticmethod
    def normcdf(x: Expression) -> Expression:
        """
        Return the elementwise standard normal CDF of the input expression x.

        This lowers to Thor's NORMCDF expression op, which is emitted with CUDA's built-in normcdf implementation.
        """

    @staticmethod
    def sigmoid(x: Expression) -> Expression: ...

    @staticmethod
    def softplus(x: Expression) -> Expression: ...

    @staticmethod
    def elu(x: Expression, alpha: float = 1.0) -> Expression: ...

    @staticmethod
    def selu(x: Expression) -> Expression: ...

    @staticmethod
    def gelu(x: Expression) -> Expression: ...

    @staticmethod
    def mish(x: Expression) -> Expression: ...

    @staticmethod
    def relu6(x: Expression) -> Expression: ...

    @staticmethod
    def hard_tanh(x: Expression, min_value: float = -1.0, max_value: float = 1.0) -> Expression: ...

    @staticmethod
    def hard_swish(x: Expression) -> Expression: ...

    @staticmethod
    def threshold(x: Expression, threshold: float = 0.0, value: float = 0.0) -> Expression: ...

    @staticmethod
    def swish(x: Expression) -> Expression: ...

    @staticmethod
    def softmax(x: Expression, algorithm: str = 'accurate', mode: str = 'channel') -> Expression:
        """
        Return cuDNN softmax of the input expression x.

        algorithm may be 'accurate' (default) or 'fast'. Log-softmax is a different operation; use log_softmax().
        mode may be 'channel' (default) or 'instance'.
        """

    @staticmethod
    def log_softmax(x: Expression, mode: str = 'channel') -> Expression:
        """Return cuDNN log-softmax of the input expression x."""

    @staticmethod
    def unsqueeze(x: Expression, axis: object) -> Expression:
        """
        Insert singleton dimensions at the specified output axes.

        Parameters
        ----------
        x : thor.physical.Expression
            Input expression.
        axis : int | list[int]
            Output-axis positions at which singleton dimensions of size 1 are inserted.

        Returns
        -------
        thor.physical.Expression
            An Expression that views the same logical values with the requested singleton axes inserted.
        """

    @staticmethod
    def squeeze(x: Expression, axis: object | None = None) -> Expression:
        """
        Remove singleton dimensions at the specified axes.

        Parameters
        ----------
        x : thor.physical.Expression
            Input expression.
        axis : int | list[int]
            Axes that must be singleton dimensions of size 1.

        Returns
        -------
        thor.physical.Expression
            An Expression that views the same logical values with the requested singleton axes removed.
        """

    @staticmethod
    def reduce_sum(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by summation across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def reduce_prod(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by product across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def reduce_min(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by minimum across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def reduce_max(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by maximum across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def argmin(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.uint32) -> Expression:
        """
        Return the flattened index of the minimum across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.uint32
                The flattened reduced-space index dtype written back to memory. Currently only uint32 is supported.
        """

    @staticmethod
    def argmax(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.uint32) -> Expression:
        """
        Return the flattened index of the maximum across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.uint32
                The flattened reduced-space index dtype written back to memory. Currently only uint32 is supported.
        """

    @staticmethod
    def reduce_mean(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by arithmetic mean across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def reduce_norm1(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by L1 norm across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def reduce_norm2(expr: Expression, axis: object | None = None, squeeze: object = False, compute_dtype: thor.DataType | None = thor.DataType.fp32, output_dtype: thor.DataType | None = thor.DataType.fp32) -> Expression:
        """
        Reduce by L2 norm across the specified axes.

        Args:
            axis: int | list[int] | None
                Single axis or sequence of axes to reduce. If None, reduce across all axes.
            squeeze: bool | int | list[int]
                If False, keep reduced axes as singleton dimensions.
                If True, remove all singleton dimensions after reduction.
                If an int or sequence of ints, remove those specific singleton axes after reduction.
            compute_dtype: thor.DataType: default thor.DataType.fp32
                The data type used during compute. Currently only fp32 is supported for this operation.
            output_dtype: thor.DataType: default thor.DataType.fp32
                Reduction stages always materialize fp32, regardless of input dtype. Add an explicit cast after the reduction to narrow it.
        """

    @staticmethod
    def outputs(outputs: dict) -> Outputs:
        """
        Create a terminal multi-output graph from a mapping of output names to expressions.

        Args:
            outputs: dict[str, Expression]
                Mapping from output names to expressions. All expressions must belong to the same graph.

        Returns:
            Outputs
                A terminal multi-output graph object that can be compiled together.
        """

    @staticmethod
    def if_else(predicate: Expression, then_outputs: Outputs, else_outputs: Outputs) -> Outputs:
        """
        Create graph-level conditional outputs from a scalar BOOLEAN predicate and two branch Outputs objects.
        """

    @staticmethod
    def if_elif_else(predicate: Expression, then_outputs: Outputs, elif_branches: Sequence[tuple[Expression, Outputs]], else_outputs: Outputs) -> Outputs:
        """
        Create graph-level conditional outputs with ordered elif branches.

        Each item in elif_branches is a (predicate, outputs) pair. Predicates are evaluated in
        order and only the first matching branch executes.
        """

    @overload
    @staticmethod
    def compile(expr: Expression, device_num: int = 0) -> FusedEquation:
        """
        Compile an expression into a fused equation.

        Parameters
        ----------
        expr : thor.physical.Expression
            The expression to compile.
        device_num : int, default 0
            The GPU device number.

        Returns
        -------
        thor.physical.FusedEquation
            The compiled fused equation.
        """

    @overload
    @staticmethod
    def compile(expr: Outputs, device_num: int = 0) -> FusedEquation: ...

class Outputs:
    def compile(self, device_num: int = 0) -> FusedEquation: ...

    def to_json(self) -> str: ...

    @staticmethod
    def from_json(payload: str, allow_unsafe_loaded_cuda_kernel_source: bool = False, trusted_cuda_kernel_public_key: str = '', trusted_cuda_kernel_source_decryption_key: str = '') -> Outputs: ...

    def output_names(self) -> list[str]: ...

    @staticmethod
    def conditional(predicate: Expression, then_outputs: Outputs, else_outputs: Outputs) -> Outputs:
        """
        Create graph-level conditional outputs. The predicate must evaluate to a single BOOLEAN element.
        Only the selected branch is executed at runtime; both branches must expose identical output names.
        """

    @staticmethod
    def if_else(predicate: Expression, then_outputs: Outputs, else_outputs: Outputs) -> Outputs: ...

    @staticmethod
    def if_elif_else(predicate: Expression, then_outputs: Outputs, elif_branches: Sequence[tuple[Expression, Outputs]], else_outputs: Outputs) -> Outputs:
        """
        Create graph-level conditional outputs with one or more ordered elif branches.
        Each elif branch is a (predicate, outputs) pair. Predicates are evaluated in order and
        only the first matching branch executes; all branches must expose identical output names.
        """

class ExpressionDefinition:
    @staticmethod
    def from_outputs(outputs: Outputs) -> ExpressionDefinition: ...

    def to_json(self) -> str: ...

    @staticmethod
    def from_json(payload: str, allow_unsafe_loaded_cuda_kernel_source: bool = False, trusted_cuda_kernel_public_key: str = '', trusted_cuda_kernel_source_decryption_key: str = '') -> ExpressionDefinition: ...

    def cuda_kernel_source_info(self) -> list: ...

    def cuda_kernel_sources(self) -> list[str]: ...

    def cuda_kernel_source_info_json(self) -> str: ...

    def cuda_kernel_signing_public_keys(self) -> list[str]: ...

    def cuda_kernel_out_of_band_keys(self) -> list: ...

    def allow_unsafe_loaded_cuda_kernel_source_compilation(self, trusted_cuda_kernel_public_key: str, trusted_cuda_kernel_source_decryption_key: str = '') -> None: ...

    @property
    def has_cuda_kernel_expressions(self) -> bool: ...

    @property
    def expected_input_names(self) -> list[str]: ...

    @property
    def expected_output_names(self) -> list[str]: ...

    @property
    def canonical_hash(self) -> str: ...

class FusedEquation:
    @overload
    def stamp(self, inputs: Mapping[str, PhysicalTensor], stream: Stream, *, tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_output: PhysicalTensor | None = None, requested_output_shape: Sequence[int] | None = []) -> Equation:
        """
        Create an executable instance of a fused equation.

        Parameters
        ----------
        inputs : dict[str, PhysicalTensor]
            Mapping from input names to tensors.
        stream : thor.Stream
            Stream on which to stamp the equation.
        tensor_scalar_inputs : dict[str, TensorScalarBinding], optional
            GPU-backed runtime scalar bindings.
        preallocated_output : PhysicalTensor | None, optional
            Preallocated output tensor for the single output.
        requested_output_shape : list[int], optional
            Requested output shape for the single output.

        Returns
        -------
        thor.physical.Equation
            A stamped execution plan.
        """

    @overload
    def stamp(self, inputs: Mapping[str, PhysicalTensor], stream: Stream, *, tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_outputs: Mapping[str, PhysicalTensor] = {}, requested_output_shapes: Mapping[str, Sequence[int]] = {}) -> Equation:
        """
        Create an executable instance of a multi-output fused equation.

        Parameters
        ----------
        inputs : dict[str, PhysicalTensor]
            Mapping from input names to tensors.
        stream : thor.Stream
            Stream on which to stamp the equation.
        tensor_scalar_inputs : dict[str, TensorScalarBinding], optional
            GPU-backed runtime scalar bindings.
        preallocated_outputs : dict[str, PhysicalTensor], optional
            Mapping from output names to preallocated output tensors.
        requested_output_shapes : dict[str, list[int]], optional
            Mapping from output names to requested output shapes.

        Returns
        -------
        thor.physical.Equation
            A stamped execution plan.
        """

    @overload
    def run(self, input: PhysicalTensor, output: PhysicalTensor, stream: Stream) -> None:
        """
        Run a fused equation with the thor.physical.PhysicalTensor's provided.

        input: PhysicalTensor
        output: PhysicalTensor
        """

    @overload
    def run(self, inputs: Mapping[str, PhysicalTensor], scalar_inputs: Mapping[str, float], output: PhysicalTensor, stream: Stream) -> None:
        """Run a fused equation with bound tensor and runtime scalar inputs."""

    @overload
    def run(self, inputs: Mapping[str, PhysicalTensor], output: PhysicalTensor, stream: Stream) -> None:
        """
        Run a fused equation with the thor.physical.PhysicalTensor's provided.

        inputs: dict[str, PhysicalTensor]
            A dict mapping input names to tensors
        output: PhysicalTensor
        """

    @overload
    def run(self, input: PhysicalTensor, outputs: Mapping[str, PhysicalTensor], stream: Stream) -> None:
        """
        Run a fused equation with the thor.physical.PhysicalTensor's provided.

        input: PhysicalTensor
        outputs: dict[str, PhysicalTensor]
            A dict mapping output names to tensors
        """

    @overload
    def run(self, inputs: Mapping[str, PhysicalTensor], scalar_inputs: Mapping[str, float], outputs: Mapping[str, PhysicalTensor], stream: Stream) -> None:
        """Run a fused equation with bound tensor and runtime scalar inputs."""

    @overload
    def run(self, inputs: Mapping[str, PhysicalTensor], outputs: Mapping[str, PhysicalTensor], stream: Stream) -> None:
        """
        Run a fused equation with the thor.physical.PhysicalTensor's provided.

        inputs: dict[str, PhysicalTensor]
            A dict mapping input names to tensors
        outputs: dict[str, PhysicalTensor]
            A dict mapping output names to tensors
        """

    def get_parameter_fan_overrides(self, inputs: Mapping[str, PhysicalTensor], parameter_names: Sequence[str], *, tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, requested_output_shapes: Mapping[str, Sequence[int]] = {}) -> dict:
        """
        Infer parameter initializer fan-in/fan-out overrides for the named parameter inputs.

        Returns
        -------
        dict[str, dict[str, int]]
            Mapping from parameter name to {"fan_in": int, "fan_out": int}.
        """

    @overload
    def compile_backward(self, wrt_names: Sequence[str] = [], error_input_name: str | None = None, accumulate_grad_outputs: bool | None = False) -> FusedEquation:
        """
        Compile a backward equation for a single-output forward equation.

        The compiled backward equation expects an additional input tensor named by
        error_input_name, whose shape is compatible with the forward output.

        Args:
            wrt_names: list[str]
                Input names to differentiate with respect to. If omitted, all forward
                root inputs are differentiated and need to be supplied to the backward
                expression.
            error_input_name: str | None
                Name for the upstream-gradient input tensor.
                I.e. the incoming error gradient (for the backward computation) from the
                layer downstream in the forward direction.
        """

    @overload
    def compile_backward(self, wrt_names: Sequence[str], feature_output_name_to_error_input_name: Mapping[str, str], accumulate_grad_outputs: bool = False) -> FusedEquation:
        """
        Compile a backward equation for a multi-output forward equation.

        This overload makes the upstream gradient explicit for each named forward
        output. The compiled backward equation will expect one additional input tensor
        per entry in ``feature_output_name_to_error_input_name``.

        Args:
            wrt_names: list[str]
                Input names to differentiate with respect to.
            feature_output_name_to_error_input_name: dict[str, str]
                Mapping from forward output name to the input name that should carry the
                corresponding upstream gradient tensor.

        For example:

        bwd = fwd.compile_backward(
            ["x", "w"],
            {
                "main": "__grad_main",
                "aux": "__grad_aux",
            },
        )

        Then the backward equation will supply x_grad and w_grad as outputs.

        In the case that w is frozen and you don't want the gradient with respect to w,
        you would instead do:

        bwd = fwd.compile_backward(
            ["x"],
            {
                "main": "__grad_main",
                "aux": "__grad_aux",
            },
        )

        Then the backward equation will only supply x_grad as an ouput. When either
        __grad_main or __grad_aux does not participate in the gradient computation
        for x, the unused tensor will not be accessed - it will be ignored in that case.
        """

    def output_names(self) -> list[str]:
        """
        Returns
        -------
        list[int]
            A list of names of the outputs from this equation.
        """

    @overload
    def output_shape(self, input: PhysicalTensor) -> list[int]:
        """
        Get the shape of the output tensor for this equation, from the input tensors.

        Parameters
        ----------
        inputs: dict[str, PhysicalTensor]
            A dict mapping input names to tensors

        Returns
        -------
        list[int]
            The output tensor dimensions.
        """

    @overload
    def output_shape(self, inputs: Mapping[str, PhysicalTensor]) -> list[int]: ...

    @overload
    def output_shapes(self, input: PhysicalTensor) -> dict[str, list[int]]:
        """
        Get the shape of the output tensor for this equation, from the input tensors.

        Parameters
        ----------
        inputs: dict[str, PhysicalTensor]
            A dict mapping input names to tensors

        Returns
        -------
        dict[str, list[int]]
            output name -> tensor dimensions.
        """

    @overload
    def output_shapes(self, inputs: Mapping[str, PhysicalTensor]) -> dict[str, list[int]]: ...

class Equation:
    @overload
    def run(self) -> None:
        """Execute the stamped fused equation on the bound tensors."""

    @overload
    def run(self, runtime_scalars: Mapping[str, float]) -> None:
        """
        Execute the stamped fused equation on the bound tensors, overriding any bound runtime scalar values for this run.
        """

    @overload
    def output(self) -> PhysicalTensor:
        """
        Return the output tensor owned by this equation instance. Valid when the equation has a single output tensor.
        """

    @overload
    def output(self, name: str) -> PhysicalTensor:
        """
        Return a named output tensor from a stamped multi-output execution plan.
        """

    def outputs(self) -> dict[str, PhysicalTensor]:
        """
        Return a dict of named output tensor from a stamped multi-output execution plan.
        """

    def output_names(self) -> list[str]: ...

    def flop_count(self) -> int:
        """
        Return the semantic floating-point operation count represented by this stamped execution plan.

        Conventions:
        - elementwise arithmetic/transcendentals count as 1 op per output element
        - GEMM / matmul / convolution use 2 FLOPs per multiply-accumulate
        - shape-only ops and transpose count as 0
        - this is a semantic model FLOP count, not a backend-instruction count
        """

    def stage_flop_counts(self) -> list[int]:
        """
        Return the per-stage semantic FLOP counts for this stamped execution plan.
        """

class DynamicExpressionVariant:
    def __init__(self, equation: FusedEquation, tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, supports_backward: bool = False) -> None:
        """
        Describe an alternate dynamic-expression execution variant.

        Variants share the primary build's tensor inputs and output bindings while
        providing their own equation and tensor-scalar runtime bindings. Set
        ``supports_backward`` only when the variant's exact forward equation may be
        used for training and differentiated by ``CustomLayer``.
        """

    @property
    def equation(self) -> FusedEquation: ...

    @equation.setter
    def equation(self, arg: FusedEquation, /) -> None: ...

    @property
    def tensor_scalar_inputs(self) -> dict[str, thor._thor.physical.TensorScalarBinding]: ...

    @tensor_scalar_inputs.setter
    def tensor_scalar_inputs(self, arg: Mapping[str, thor._thor.physical.TensorScalarBinding], /) -> None: ...

    @property
    def supports_backward(self) -> bool: ...

    @supports_backward.setter
    def supports_backward(self, arg: bool, /) -> None: ...

class DynamicExpressionBuild:
    def __init__(self, equation: FusedEquation, stamp_inputs: Mapping[str, PhysicalTensor], tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_outputs: Mapping[str, PhysicalTensor] = {}, requested_output_shapes: Mapping[str, Sequence[int]] = {}, serialized_definition: ExpressionDefinition | None = None, execution_variants: Mapping[int, DynamicExpressionVariant] = {}, evaluation_variant_id: int | None = None) -> None:
        """
        Describe a prepared dynamic-expression build result.

        Parameters
        ----------
        equation : thor.physical.FusedEquation
            The compiled equation to stamp.
        stamp_inputs : dict[str, PhysicalTensor]
            Input tensors bound into the prepared expression.
        tensor_scalar_inputs : dict[str, thor.physical.TensorScalarBinding], optional
            Tensor-backed scalar bindings.
        preallocated_outputs : dict[str, PhysicalTensor], optional
            Output tensors to bind when stamping.
        requested_output_shapes : dict[str, list[int]], optional
            Per-output requested shapes used when stamping.
        serialized_definition : thor.physical.ExpressionDefinition, optional
            Serializable expression graph matching ``equation``. Provide this when an
            arbitrary dynamic builder should also support architecture serialization.
        execution_variants : dict[int, thor.physical.DynamicExpressionVariant], optional
            Alternate execution variants keyed by nonzero opaque IDs.
        evaluation_variant_id : int, optional
            Variant selected for validation and inference passes.
        """

    @property
    def equation(self) -> FusedEquation: ...

    @equation.setter
    def equation(self, arg: FusedEquation, /) -> None: ...

    @property
    def stamp_inputs(self) -> dict[str, PhysicalTensor]: ...

    @stamp_inputs.setter
    def stamp_inputs(self, arg: Mapping[str, PhysicalTensor], /) -> None: ...

    @property
    def tensor_scalar_inputs(self) -> dict[str, thor._thor.physical.TensorScalarBinding]: ...

    @tensor_scalar_inputs.setter
    def tensor_scalar_inputs(self, arg: Mapping[str, thor._thor.physical.TensorScalarBinding], /) -> None: ...

    @property
    def preallocated_outputs(self) -> dict[str, PhysicalTensor]: ...

    @preallocated_outputs.setter
    def preallocated_outputs(self, arg: Mapping[str, PhysicalTensor], /) -> None: ...

    @property
    def requested_output_shapes(self) -> dict[str, list[int]]: ...

    @requested_output_shapes.setter
    def requested_output_shapes(self, arg: Mapping[str, Sequence[int]], /) -> None: ...

    @property
    def serialized_definition(self) -> ExpressionDefinition: ...

    @serialized_definition.setter
    def serialized_definition(self, arg: ExpressionDefinition, /) -> None: ...

    @property
    def execution_variants(self) -> dict[int, DynamicExpressionVariant]: ...

    @execution_variants.setter
    def execution_variants(self, arg: Mapping[int, DynamicExpressionVariant], /) -> None: ...

    @property
    def evaluation_variant_id(self) -> int | None: ...

    @evaluation_variant_id.setter
    def evaluation_variant_id(self, arg: int) -> None: ...

class PreparedDynamicExpression:
    @overload
    def stamp(self) -> Equation:
        """
        Stamp the prepared dynamic expression using its bound inputs and any default
        preallocated outputs captured in the DynamicExpressionBuild.
        """

    @overload
    def stamp(self, preallocated_outputs_override: Mapping[str, PhysicalTensor], requested_output_shapes_override: Mapping[str, Sequence[int]] = {}) -> Equation:
        """
        Stamp the prepared dynamic expression, overriding default preallocated outputs
        and/or requested output shapes for this stamp.
        """

    def has_execution_variant(self, variant_id: int) -> bool:
        """Return whether the requested execution variant is defined."""

    def execution_variant_ids(self) -> list[int]:
        """Return the primary and alternate execution-variant IDs."""

    def evaluation_variant_id(self) -> int | None:
        """Return the configured evaluation execution-variant ID, if any."""

    def execution_variant_supports_backward(self, variant_id: int) -> bool:
        """
        Return whether the variant may be used for matching forward/backward training.
        """

    def stamp_execution_variant(self, variant_id: int, preallocated_outputs_override: Mapping[str, PhysicalTensor] = {}, requested_output_shapes_override: Mapping[str, Sequence[int]] = {}) -> Equation:
        """Stamp a specific primary or alternate execution variant."""

    @overload
    def compile_backward(self, wrt_names: Sequence[str] = [], upstream_input_name: str | None = None, accumulate_grad_outputs: bool | None = False, additional_inputs: Mapping[str, PhysicalTensor] = {}, additional_tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_grad_outputs: Mapping[str, PhysicalTensor] = {}, requested_grad_output_shapes: Mapping[str, Sequence[int]] = {}) -> PreparedDynamicExpression:
        """
        Prepare a backward dynamic expression for a single-output forward expression.
        """

    @overload
    def compile_backward(self, wrt_names: Sequence[str], upstream_input_names_by_output: Mapping[str, str], accumulate_grad_outputs: bool = False, additional_inputs: Mapping[str, PhysicalTensor] = {}, additional_tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_grad_outputs: Mapping[str, PhysicalTensor] = {}, requested_grad_output_shapes: Mapping[str, Sequence[int]] = {}) -> PreparedDynamicExpression:
        """
        Prepare a backward dynamic expression for a multi-output forward expression.
        """

    def get_parameter_fan_overrides(self, parameter_names: Sequence[str]) -> dict:
        """
        Infer parameter initializer fan-in/fan-out overrides for the prepared dynamic expression.
        """

    @overload
    def stamp_backward(self, wrt_names: Sequence[str] = [], upstream_input_name: str | None = None, accumulate_grad_outputs: bool | None = False, additional_inputs: Mapping[str, PhysicalTensor] = {}, additional_tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_grad_outputs: Mapping[str, PhysicalTensor] = {}, requested_grad_output_shapes: Mapping[str, Sequence[int]] = {}) -> Equation:
        """
        Stamp a backward execution plan for a single-output forward expression.
        """

    @overload
    def stamp_backward(self, wrt_names: Sequence[str], upstream_input_names_by_output: Mapping[str, str], accumulate_grad_outputs: bool = False, additional_inputs: Mapping[str, PhysicalTensor] = {}, additional_tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_grad_outputs: Mapping[str, PhysicalTensor] = {}, requested_grad_output_shapes: Mapping[str, Sequence[int]] = {}) -> Equation:
        """Stamp a backward execution plan for a multi-output forward expression."""

    @property
    def equation(self) -> FusedEquation:
        """
        Return the compiled equation owned by this prepared dynamic expression.
        """

    @property
    def stamp_inputs(self) -> dict[str, PhysicalTensor]: ...

    @property
    def tensor_scalar_inputs(self) -> dict[str, thor._thor.physical.TensorScalarBinding]: ...

    @property
    def preallocated_outputs(self) -> dict[str, PhysicalTensor]: ...

    @property
    def requested_output_shapes(self) -> dict[str, list[int]]: ...

class DynamicExpression:
    def __init__(self, builder: Callable) -> None:
        """
        Create a dynamic expression from a Python callable.

        Parameters
        ----------
        builder : Callable[
            [dict[str, PhysicalTensor], dict[str, PhysicalTensor], thor.Stream],
            thor.physical.DynamicExpressionBuild
        ]
            A callable that receives three arguments:

            - ``inputs``:
              A mapping from input name to the currently bound input tensor.

              These are the tensors that the caller supplied when preparing or
              stamping the dynamic expression. The builder may inspect their shape,
              dtype, placement, and other metadata in order to choose how to build
              the expression.

            - ``outputs``:
              A mapping from output name to caller-supplied output tensors.

              This mapping represents the final outputs that the caller would like
              the dynamic expression to write into.

              The builder may use ``outputs`` to:

              - validate that a provided output tensor has the expected shape, dtype,
                or placement
              - choose an output dtype or architecture that is compatible with the
                requested outputs
              - pass those tensors through as ``preallocated_outputs`` in the returned
                ``DynamicExpressionBuild`` so the compiled equation writes into them

              In other words, ``outputs`` is part of the builder's decision surface.
              It is not just informational metadata.

            - ``stream``:
              The stream that will be used for preparation / stamping / execution.

              The builder may inspect this if needed, but it should generally use the
              stream only as contextual information when constructing the returned
              ``DynamicExpressionBuild``.

        Returns
        -------
        thor.physical.DynamicExpressionBuild
            An object describing the compiled equation and any default bindings to use
            when stamping. Typically this includes:

            - a compiled equation
            - the chosen input bindings
            - any caller-provided output tensors that should be reused as final outputs

        Notes
        -----
        The callable is invoked synchronously from C++ when ``prepare(...)``,
        ``stamp(...)``, or ``stamp_backward(...)`` is called.

        This means the builder runs on the preparation / stamping path, not on the
        hot kernel execution path. It is therefore appropriate for the builder to make
        shape-dependent, dtype-dependent, or architecture-dependent decisions.

        The builder is expected to *describe* the computation by returning a
        ``DynamicExpressionBuild``. It should not directly enqueue the actual forward
        or backward kernels itself.

        Examples
        --------
        A simple example that chooses the expression based on the shape of the input:

        .. code-block:: python

            import thor
            from thor.physical import Expression as ex
            from thor.physical import DynamicExpression, DynamicExpressionBuild, FusedEquation

            def make_dynamic_relu(device_num: int):
                def builder(inputs, outputs, stream):
                    x_tensor = inputs["x"]

                    if len(x_tensor.dims) != 2:
                        raise ValueError("expected a rank-2 input")

                    x = ex.input("x")
                    y = ex.max(x, 0.0)

                    named = ex.outputs({"y": y})
                    equation = FusedEquation.compile(named.physical_outputs(), device_num)

                    return DynamicExpressionBuild(
                        equation=equation,
                        stamp_inputs=inputs,
                        preallocated_outputs=outputs,
                    )

                return DynamicExpression(builder)

        Using ``outputs`` to validate and reuse a caller-provided output tensor:

        .. code-block:: python

            import thor
            from thor.physical import Expression as ex
            from thor.physical import DynamicExpression, DynamicExpressionBuild, FusedEquation

            def make_dynamic_identity(device_num: int):
                def builder(inputs, outputs, stream):
                    x_tensor = inputs["x"]

                    if "y" in outputs:
                        y_tensor = outputs["y"]
                        if y_tensor.dims != x_tensor.dims:
                            raise ValueError(
                                f"output y has dims {y_tensor.dims}, expected {x_tensor.dims}"
                            )
                        if y_tensor.placement != x_tensor.placement:
                            raise ValueError("output y must be on the same device as x")

                    x = ex.input("x")
                    named = ex.outputs({"y": x})
                    equation = FusedEquation.compile(named.physical_outputs(), device_num)

                    return DynamicExpressionBuild(
                        equation=equation,
                        stamp_inputs=inputs,
                        preallocated_outputs=outputs,
                    )

                return DynamicExpression(builder)

        A fully connected layer can be expressed by inspecting the bound parameter
        tensors at build time:

        .. code-block:: python

            import thor
            from thor.physical import Expression as ex
            from thor.physical import DynamicExpression, DynamicExpressionBuild, FusedEquation

            def fully_connected_dynamic_expression(device_num: int, has_bias: bool = True):
                def builder(inputs, outputs, stream):
                    x_tensor = inputs["feature_input"]
                    w_tensor = inputs["weights"]

                    if len(x_tensor.dims) != 2:
                        raise ValueError("feature_input must be rank 2")
                    if len(w_tensor.dims) != 2:
                        raise ValueError("weights must be rank 2")
                    if x_tensor.dims[1] != w_tensor.dims[0]:
                        raise ValueError("feature_input and weights have incompatible shapes")

                    x = ex.input("feature_input")
                    w = ex.input("weights", output_dtype=w_tensor.dtype, compute_dtype=w_tensor.dtype)
                    y = x @ w

                    if has_bias:
                        b_tensor = inputs["biases"]
                        if len(b_tensor.dims) != 1 or b_tensor.dims[0] != w_tensor.dims[1]:
                            raise ValueError("biases must have shape [out_features]")
                        b = ex.input("biases", output_dtype=b_tensor.dtype, compute_dtype=b_tensor.dtype)
                        y = y + b

                    named = ex.outputs({"feature_output": y})
                    equation = FusedEquation.compile(named.physical_outputs(), device_num)

                    return DynamicExpressionBuild(
                        equation=equation,
                        stamp_inputs=inputs,
                        preallocated_outputs=outputs,
                    )

                return DynamicExpression(builder)
        """

    @staticmethod
    def from_expression_definition(definition: ExpressionDefinition) -> DynamicExpression: ...

    @property
    def serialized_definition(self) -> ExpressionDefinition: ...

    def prepare(self, inputs: Mapping[str, PhysicalTensor], outputs: Mapping[str, PhysicalTensor], stream: Stream) -> PreparedDynamicExpression:
        """
        Validate the provided tensors and stream, invoke the Python builder with
        ``(inputs, outputs, stream)``, and return a PreparedDynamicExpression.
        """

    @overload
    def stamp(self, inputs: Mapping[str, PhysicalTensor], outputs: Mapping[str, PhysicalTensor], stream: Stream) -> Equation:
        """
        Validate the provided tensors and stream, then stamp the dynamic expression.
        """

    @overload
    def stamp(self, inputs: Mapping[str, PhysicalTensor], outputs: Mapping[str, PhysicalTensor], stream: Stream, preallocated_outputs: Mapping[str, PhysicalTensor], requested_output_shapes: Mapping[str, Sequence[int]] = {}) -> Equation:
        """
        Validate the provided tensors and stream, then stamp the dynamic expression
        with output overrides.
        """

    @overload
    def stamp_backward(self, inputs: Mapping[str, PhysicalTensor], outputs: Mapping[str, PhysicalTensor], stream: Stream, wrt_names: Sequence[str] = [], upstream_input_name: str | None = None, accumulate_grad_outputs: bool | None = False, additional_inputs: Mapping[str, PhysicalTensor] = {}, additional_tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_grad_outputs: Mapping[str, PhysicalTensor] = {}, requested_grad_output_shapes: Mapping[str, Sequence[int]] = {}) -> Equation:
        """
        Prepare and stamp a backward execution plan for a single-output forward expression.
        """

    @overload
    def stamp_backward(self, inputs: Mapping[str, PhysicalTensor], outputs: Mapping[str, PhysicalTensor], stream: Stream, wrt_names: Sequence[str], upstream_input_names_by_output: Mapping[str, str], accumulate_grad_outputs: bool = False, additional_inputs: Mapping[str, PhysicalTensor] = {}, additional_tensor_scalar_inputs: Mapping[str, thor._thor.physical.TensorScalarBinding] = {}, preallocated_grad_outputs: Mapping[str, PhysicalTensor] = {}, requested_grad_output_shapes: Mapping[str, Sequence[int]] = {}) -> Equation:
        """
        Prepare and stamp a backward execution plan for a multi-output forward expression.
        """
