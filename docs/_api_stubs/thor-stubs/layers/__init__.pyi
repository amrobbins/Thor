from collections.abc import Sequence
import enum
from typing import overload

import thor
from thor._thor.layers import (
    RaggedNetworkInput as RaggedNetworkInput,
    TypeConverter as TypeConverter
)
import thor.activations
import thor.initializers
import thor.optimizers
import thor.parameters
import thor.physical
from thor.physical import (
    CudaKernelExpression as _CudaKernelExpression
)
import thor.runtime


class Layer:
    def get_id(self) -> int: ...

class MultiConnectionLayer(Layer):
    pass

class TrainableLayer(MultiConnectionLayer):
    def freeze_training(self) -> None: ...

    def unfreeze_training(self) -> None: ...

    def is_training_frozen(self) -> bool: ...

    def get_parameters(self) -> list[thor.parameters.ParameterSpecification]: ...

    def get_bound_parameter(self, placed_network: thor.runtime.PlacedNetwork, name: str) -> thor.parameters.BoundParameter: ...

    def get_bound_parameters(self, placed_network: thor.runtime.PlacedNetwork) -> list[thor.parameters.BoundParameter]: ...

    def get_parameter_reference(self, name: str) -> thor.parameters.ParameterReference: ...

    def get_parameter_references(self, trainable_only: bool = True, training_enabled_only: bool = True) -> list[thor.parameters.ParameterReference]: ...

class TensorSpec:
    def __init__(self, shape: Sequence[int], dtype: thor.DataType) -> None: ...

    @property
    def shape(self) -> list[int]: ...

    @shape.setter
    def shape(self, arg: Sequence[int], /) -> None: ...

    @property
    def dtype(self) -> thor.DataType: ...

    @dtype.setter
    def dtype(self, arg: thor.DataType, /) -> None: ...

class CustomLayerSpecContext:
    def input_spec(self, name: str) -> TensorSpec: ...

    @property
    def inputs(self) -> dict[str, TensorSpec]: ...

class CustomLayerBuildContext:
    @property
    def inputs(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def input_tensors(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def parameters(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def parameter_tensors(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def param_tensors(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def outputs(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def output_tensors(self) -> dict[str, thor.physical.PhysicalTensor]: ...

    @property
    def stream(self) -> thor.physical.Stream: ...

    @property
    def device_num(self) -> int: ...

    def input_tensor(self, name: str) -> thor.physical.PhysicalTensor: ...

    def parameter_tensor(self, name: str) -> thor.physical.PhysicalTensor: ...

    def param_tensor(self, name: str) -> thor.physical.PhysicalTensor: ...

    def output_tensor(self, name: str) -> thor.physical.PhysicalTensor: ...

    def has_input(self, name: str) -> bool: ...

    def has_parameter(self, name: str) -> bool: ...

    def has_param(self, name: str) -> bool: ...

    def has_output(self, name: str) -> bool: ...

    def input(self, name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression: ...

    def param(self, name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression: ...

class CustomLayer(TrainableLayer):
    def __init__(self, network: thor.Network, inputs: object, output_names: object | None = None, build: object | None = None, parameters: object | None = None, optimizer: thor.optimizers.Optimizer | None = None, activation: thor.activations.Activation | None = None, output_specs: object | None = None, uses_batch_validity: bool = False, requires_full_batch: bool = False) -> None:
        """
        Python-facing CustomLayer.

        The C++ API layer owns the CustomLayer construction logic, including named input/output
        interfaces and logical output tensor inference. Python can use it either directly by
        passing build=..., output_specs=..., and parameters=..., or by subclassing and overriding
        output_specs(context), parameters(), and build(context).

        The build argument may be either a Python callable or a thor.physical.DynamicExpression.
        When output_specs is supplied, API output-shape construction uses only logical, batch-free
        TensorSpec values. Before the layer is accepted into the model, Thor probes the physical
        builder to require a pure tensor expression whose ExpressionDefinition is identical across
        batch probes or can be safely generalized to internal symbolic dimensions. Builders with
        pre-forward callbacks, runtime scalar bindings, or other non-declarative state are rejected
        during CustomLayer construction; generic Python CustomLayer instances are never runtime-only.
        CudaKernelExpression.as_dynamic_expression() remains serializable when its signed
        ExpressionDefinition is already batch-polymorphic. Terminal storage aliases such as
        strided_view are materialized generically into the layer's dense public output tensor.

        Convenience forms:
        - inputs=<thor.Tensor or thor.RaggedTensor> defaults to {"feature_input": tensor}
        - output_names omitted defaults to ["feature_output"]
        - ragged inputs produce partition-preserving thor.RaggedTensor outputs; all named ragged inputs must share one offsets tensor
        - activation=<thor.activations.Activation> stitches that activation onto each returned expression before compilation
        - uses_batch_validity=True declares runtime batch-validity use; Thor currently exposes it as
          ``thor.BATCH_VALIDITY_MASK_NAME`` through ``context.input(...)``
        - requires_full_batch=True rejects partial-tail submissions for batch-coupled expressions that do not implement masked semantics
        """

    def parameters(self) -> list: ...

    def output_specs(self, context: CustomLayerSpecContext) -> object: ...

    def build(self, context: CustomLayerBuildContext) -> dict: ...

    @property
    def uses_batch_validity(self) -> bool: ...

    @property
    def requires_full_batch(self) -> bool: ...

    @property
    def use_ragged(self) -> bool: ...

    def get_use_ragged(self) -> bool: ...

    def get_input_interface(self, interface_index: int = 0) -> object: ...

    def get_output_interface(self, inputs: object) -> object: ...

    def get_output_interface_by_index(self, interface_index: int = 0) -> object: ...

    def get_output(self, name: str, interface_index: int = 0) -> object: ...

    def __getitem__(self, name: str) -> object: ...

    @property
    def outputs(self) -> object: ...

    def get_input_names(self) -> list[str]: ...

    def get_output_names(self) -> list[str]: ...

    def get_parameters(self) -> list[thor.parameters.ParameterSpecification]: ...

    def get_bound_parameter(self, placed_network: thor.runtime.PlacedNetwork, name: str) -> thor.parameters.BoundParameter: ...

    def get_bound_parameters(self, placed_network: thor.runtime.PlacedNetwork) -> list[thor.parameters.BoundParameter]: ...

class Add(MultiConnectionLayer):
    """
    Elementwise addition for dense tensors or canonical rank-1 ragged tensors.

    Ragged operands must share the exact same row-partition offsets tensor. The
    result preserves that partition and executes only over the authoritative active
    packed prefix.
    """

    @overload
    def __init__(self, network: thor.Network, left: thor.Tensor, right: thor.Tensor) -> None: ...

    @overload
    def __init__(self, network: thor.Network, left: thor.RaggedTensor, right: thor.RaggedTensor) -> None: ...

    def get_feature_output(self) -> object: ...

    @property
    def use_ragged(self) -> bool: ...

class AdaptiveLayerNorm(MultiConnectionLayer):
    """
    Adaptive layer normalization over a contiguous trailing normalized shape.

    AdaptiveLayerNorm differs from LayerNorm by taking scale and bias as input tensors rather
    than trainable parameters. For a dense feature_input, scale and bias are per-sample affine
    values as before. For a rank-1 RaggedTensor feature_input, scale and bias are dense per-logical-
    row values and Thor broadcasts each row's affine parameters only to that row's active tokens.
    The exact ragged row partition is preserved.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor or thor.RaggedTensor
        Input feature tensor to normalize. Ragged input currently requires exactly one trailing
        channel dimension.
    scale_input : thor.Tensor
        Per-sample/per-logical-row scale tensor. API dimensions must match normalized_shape and
        dtype must be fp32.
    bias_input : thor.Tensor
        Per-sample/per-logical-row bias tensor. API dimensions must match normalized_shape and
        dtype must be fp32.
    normalized_shape : Sequence[int] or None, default None
        Trailing feature dimensions to normalize over. None normalizes the final feature dimension.
    epsilon : float, default 1e-5
        Positive numerical-stability epsilon.
    scale_bias_data_type : thor.DataType, default thor.DataType.fp32
        Data type for scale and bias tensors. Thor currently requires fp32 for cuDNN Frontend AdaLayerNorm.
    """

    def __init__(self, network: thor.Network, feature_input: object, scale_input: thor.Tensor, bias_input: thor.Tensor, normalized_shape: object | None = None, epsilon: float = 1e-05, scale_bias_data_type: thor.DataType = thor.DataType.fp32) -> None: ...

    def get_feature_output(self) -> object:
        """
        Return the output tensor produced by this layer. Ragged inputs return a thor.RaggedTensor.
        """

    def get_data_input(self) -> object: ...

    def get_scale_input(self) -> thor.Tensor: ...

    def get_bias_input(self) -> thor.Tensor: ...

    def get_normalized_shape(self) -> list[int]: ...

    def get_epsilon(self) -> float: ...

    def get_scale_bias_data_type(self) -> thor.DataType: ...

    def get_use_ragged(self) -> bool: ...

class Attention(CustomLayer):
    def __init__(self, network: thor.Network, query_input: object, key_input: object, value_input: object, num_heads: int, num_key_value_heads: int | None = None, head_dim: int | None = None, value_dim: int | None = None, output_features: int | None = None, has_bias: bool | None = False, mask_kind: str = 'none', diagonal_left_bound: int = 0, diagonal_right_bound: int = 0, use_alibi_mask: bool = False, attention_scale: float | None = None, use_rope: bool | None = False, rope_rotary_dim: int = 0, rope_base: float = 10000.0, rope_position_offset: int = 0, rope_interleaved: bool = False, rope_scaling_kind: str = 'none', rope_scaling_factor: float = 1.0, rope_original_max_position_embeddings: int = 0, rope_attention_factor: float | None = None, rope_yarn_beta_fast: float | None = 32.0, rope_yarn_beta_slow: float = 1.0, rope_llama3_low_freq_factor: float = 1.0, rope_llama3_high_freq_factor: float = 4.0, rope_long_rope_short_factors: Sequence[float] = [], rope_long_rope_long_factors: Sequence[float] = [], weights_data_type: thor.DataType | None = None, compute_data_type: thor.DataType | None = thor.DataType.fp32, output_data_type: thor.DataType | None = None, weights_initializer: thor.initializers.Initializer | None = None, bias_initializer: thor.initializers.Initializer | None = None, optimizer: thor.optimizers.Optimizer | None = None, rope_in_place: bool = False, sdpa_dropout_probability: float = 0.0, sdpa_dropout_seed: int = 0, sdpa_dropout_offset: int = 0, query_sequence_lengths: thor.Tensor | None = None, key_value_sequence_lengths: thor.Tensor | None = None, score_bias_input: thor.Tensor | None = None, epilogue: object | None = None, epilogue_inputs: object | None = None, rope_query_position_offset: int | None = None, rope_key_position_offset: int | None = None, rope_query_position_offsets: thor.Tensor | None = None, rope_key_position_offsets: thor.Tensor | None = None, output_dropout_probability: float | None = 0.0, output_dropout_seed: int | None = None, residual_input: object | None = None, dropout_probability: float | None = None, dropout_seed: int | None = None, dropout_offset: int | None = None) -> None:
        """
        Public transformer attention layer built from learned Q/K/V/O projections and the
        cuDNN scaled-dot-product attention stage.

        API tensor shapes omit batch. ``query_input`` is ``[Sq, query_features]``;
        ``key_input`` and ``value_input`` are independently supplied as ``[Skv, key_features]``
        and ``[Skv, value_features]``. K and V must share sequence geometry, but their feature
        widths and source tensors are independent. Placement adds the batch dimension, so the cuDNN hot path consumes semantic
        ``[B, H, S, D]`` tensors after projection.

        Supported production dtype surface:

        * FP16 and BF16 forward/backward are the first-class training path.  Q/K/V/O use
          one FP16 or BF16 storage dtype and attention compute/intermediate dtype is FP32.
        * FP8 attention is not exposed by this high-level learned-projection layer.  Use
          ``thor.layers.ScaledDotProductAttention`` for the experimental FP8 forward-only
          low-level SDPA path.

        Supported features for FP16/BF16:

        * Q, K, and V are independent inputs. Self-attention is expressed by passing the same
          tensor for all three; conventional cross-attention passes the decoder state as Q and
          the encoder state as both K and V.
        * MHA, GQA, and MQA: ``num_heads`` must be an integer multiple of
          ``num_key_value_heads``.
        * RoPE with ``none``, ``linear``, ``dynamic_ntk``, ``yarn``, ``longrope``, and
          ``llama3`` scaling parameterizations.
          ``rope_position_offset`` is the shared Q/K origin. Cross-attention may override it
          independently with ``rope_query_position_offset`` and ``rope_key_position_offset``.
          Ragged attention may instead provide per-row absolute origins through the INT32
          logical-[1] ``rope_query_position_offsets`` and ``rope_key_position_offsets`` inputs;
          a supplied per-row input replaces the scalar origin for that side. Q and K still
          share the same rotary basis/scaling parameters. Dynamic-NTK and LongRoPE use
          FP32 positional metadata, so absolute positions/extents and
          ``rope_original_max_position_embeddings`` must remain at most 16,777,216 for exact
          integer representation. Thor validates this statically for scalar origins; values
          supplied through per-row origin tensors must obey the same bound at runtime.
        * Masks: ``none``, ``causal_top_left``, ``causal_bottom_right``,
          ``sliding_window_top_left``, and ``sliding_window_bottom_right``.
        * ALiBi only with causal/sliding diagonal masks and ``diagonal_right_bound == 0``;
          cuDNN rejects ALiBi with a positive right bound.
        * Additive score bias via ``score_bias_input`` with logical API shape
          ``[1|num_heads, 1|Sq, 1|Skv]`` and dtype equal to ``compute_data_type``.  Forward
          supports sequence broadcast.  Backward materializes sequence-broadcast bias to
          dense score space before cuDNN backward, then reduces dBias back to the public
          bias shape.  Ragged + additive-bias backward is rejected.
        * Output-projection epilogues support the primary projected output plus named
          auxiliary tensors. Auxiliary tensors must already match the public Attention
          output shape, storage dtype, and placement; Thor does not insert conversions.
        * ``sdpa_dropout_probability``, ``sdpa_dropout_seed``, and ``sdpa_dropout_offset``
          control cuDNN Philox dropout on the SDPA probability matrix. Thor advances the
          runtime offset by ``B * Hq * Sq * Skv``. The legacy ``dropout_probability``,
          ``dropout_seed``, and ``dropout_offset`` keywords remain accepted as aliases.
        * ``output_dropout_probability`` controls dropout after the learned output projection.
          ``output_dropout_seed`` is optional; when omitted Thor chooses an independent seed
          for the layer and persists that chosen seed in the serialized architecture.
          When ``residual_input`` is supplied, the exact contract is
          ``residual_input + dropout(projected_output)`` during training. Thor fuses residual
          addition into the projection GEMM when output dropout is inactive (including
          validation/inference), and otherwise uses one native fused dropout+residual post-op.
          A residual must match the query/output domain; ragged residuals must share the exact
          query row partition.
        * Padding masks use ``query_sequence_lengths`` and ``key_value_sequence_lengths``
          together, both int32 logical ``[1]`` tensors.
        * ``query_input``, ``key_input``, and ``value_input`` accept ``thor.Tensor`` or
          ``thor.RaggedTensor``. K and V must both be dense or both ragged and must share the same
          sequence geometry; their feature widths and source tensors may differ. The output domain
          follows Q: dense Q produces dense O, while ragged Q preserves the query row partition on O.
          All four dense/ragged Q/O and K/V combinations are supported. RoPE positions reset at each packed row;
          scalar origins apply to a dense side or every row of a ragged side, while the optional
          per-row origin tensors replace the scalar origin for the corresponding ragged domain.
          Q and K need only have the same logical batch size.

        Important combination rules:

        * Bottom-right/decode masks currently require additive bias, ALiBi, and dropout
          to be disabled in the production cuDNN primary SDPA path.
        * This layer does not expose paged KV cache; use the physical expression SDPA API
          for the low-level inference-only paged-KV path.
        """

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the primary output-projection input expression expected by an Attention epilogue.
        """

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return a named auxiliary tensor input expression for an Attention epilogue.
        """

    def get_feature_output(self) -> object: ...

    def get_num_heads(self) -> int: ...

    def get_num_key_value_heads(self) -> int: ...

    def get_head_dim(self) -> int: ...

    def get_value_dim(self) -> int: ...

    def get_output_features(self) -> int: ...

    def get_has_bias(self) -> bool: ...

    def get_use_rope(self) -> bool: ...

    def get_rope_in_place(self) -> bool: ...

    def get_rope_query_position_offset(self) -> int: ...

    def get_rope_key_position_offset(self) -> int: ...

    def get_rope_query_position_offsets_input(self) -> thor.Tensor | None: ...

    def get_rope_key_position_offsets_input(self) -> thor.Tensor | None: ...

    def get_rope_scaling_kind(self) -> str: ...

    def get_rope_scaling_factor(self) -> float: ...

    def get_rope_original_max_position_embeddings(self) -> int: ...

    def get_mask_kind(self) -> str: ...

    def get_diagonal_left_bound(self) -> int: ...

    def get_diagonal_right_bound(self) -> int: ...

    def get_use_alibi_mask(self) -> bool: ...

    def get_attention_scale(self) -> float | None: ...

    def get_sdpa_dropout_probability(self) -> float: ...

    def get_dropout_probability(self) -> float: ...

    def set_training_dropout_enabled(self, enabled: bool) -> None: ...

    def is_training_dropout_enabled(self) -> bool: ...

    def get_sdpa_dropout_seed(self) -> int: ...

    def get_sdpa_dropout_offset(self) -> int: ...

    def get_dropout_seed(self) -> int: ...

    def get_dropout_offset(self) -> int: ...

    def get_output_dropout_probability(self) -> float: ...

    def get_output_dropout_seed(self) -> int: ...

    def get_use_residual(self) -> bool: ...

    def get_residual_input(self) -> object: ...

    def get_use_ragged(self) -> bool: ...

    def get_query_ragged(self) -> bool: ...

    def get_key_value_ragged(self) -> bool: ...

    def get_query_input(self) -> object: ...

    def get_key_input(self) -> object: ...

    def get_value_input(self) -> object: ...

    def get_use_score_bias(self) -> bool: ...

    def get_score_bias_input(self) -> thor.Tensor | None: ...

    def get_use_sequence_lengths(self) -> bool: ...

    def get_query_sequence_lengths_input(self) -> thor.Tensor | None: ...

    def get_key_value_sequence_lengths_input(self) -> thor.Tensor | None: ...

    def get_weights_data_type(self) -> thor.DataType: ...

    def get_compute_data_type(self) -> thor.DataType: ...

    def get_output_data_type(self) -> thor.DataType: ...

    def get_has_epilogue(self) -> bool: ...

    def get_epilogue_input_names(self) -> list[str]: ...

class BatchNormalization(TrainableLayer):
    """
    Create and attach a BatchNormalization layer to a Network.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor
        Input feature tensor for this layer.
    exponential_running_average_factor : float
        FIXME.
    epsilon : float
        FIXME.
    """

    def __init__(self, network: thor.Network, feature_input: thor.Tensor, exponential_running_average_factor: float = 0.05, epsilon: float = 9.999999747378752e-05) -> None: ...

    def get_feature_output(self) -> thor.Tensor:
        """
        Return the output tensor produced by this layer.

        Returns
        -------
        thor.Tensor
            The feature output tensor handle.
        """

    def get_exponential_running_average_factor(self) -> float | None: ...

    def get_epsilon(self) -> float | None: ...

class DropOut(Layer):
    def __init__(self, network: thor.Network, feature_input: object, drop_proportion: float) -> None:
        """
        Create and attach a DropOut layer to a Network.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_input : thor.Tensor or thor.RaggedTensor
            Dense or packed-ragged input. Ragged input preserves its row partition and applies dropout only to active packed values.
        drop_proportion : float
            Fraction of units to drop (0.0 <= p <= 1.0).
        """

    def get_feature_output(self) -> object:
        """
        Return the logical output. Ragged inputs produce a RaggedTensor with the same row partition.
        """

    def get_use_ragged(self) -> bool: ...

    def get_drop_proportion(self) -> float: ...

    def set_training_dropout_enabled(self, enabled: bool) -> None: ...

    def is_training_dropout_enabled(self) -> bool: ...

class Concatenate(Layer):
    def __init__(self, network: thor.Network, feature_inputs: object, concatenation_axis: int) -> None:
        """
        Create and attach a Concatenate layer to a Network.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        feature_inputs : list[thor.Tensor] | list[thor.RaggedTensor]
            Dense feature tensors, or ragged tensors sharing the exact same row partition.
        concatenation_axis : int
            Axis along which to concatenate the input tensors.

        For example, if your input tensors have dimensions:
            1. [2, 4, 5, 7]
            2. [2, 6, 5, 7]
            3. [2, 2, 5, 7]

        with concatenation_axis=1, then your output tensor will have dimensions:
            [2, 12, 5, 7]

        Note that all dimensions must match to perform Contcatenate, except for the concatenation_axis dimension.
        """

    def get_feature_output(self) -> object:
        """Return the concatenated dense or ragged feature output."""

    @property
    def use_ragged(self) -> bool: ...

class Convolution1d(TrainableLayer):
    """
    Trainable 1D convolution over a dense CW ``thor.Tensor`` or a logical ``thor.RaggedTensor``.

    Dense inputs retain the existing padding surface: ``"valid"``, ``"same"``
    (SAME_UPPER), ``"causal"``, or an explicit ``(left, right)`` pair. Ragged
    inputs deliberately support only stride-1 causal convolution and preserve
    the exact input row partition while changing the trailing channel count.
    The ragged path lowers to Thor's qualified ragged causal Conv1D backend; it
    does not materialize a dense tensor or an im2col/unfold temporary.
    ``compute_data_type=thor.DataType.fp32`` requests strict FP32 convolution
    math. ``thor.DataType.tf32`` explicitly permits TensorFloat-32 execution
    for FP32 input, weight, and output storage.
    """

    def __init__(self, network: thor.Network, feature_input: object, num_output_channels: int, filter_width: int, stride: int = 1, padding: object = 'valid', has_bias: bool = True, activation: object | None = '__thor_default_activation__', weights_initializer: thor.initializers.Initializer | None = None, biases_initializer: thor.initializers.Initializer | None = None, epilogue: object | None = None, epilogue_inputs: object | None = None, dilation: int = 1, groups: int = 1, compute_data_type: thor.DataType = thor.DataType.fp32) -> None: ...

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression: ...

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression: ...

    def get_feature_output(self) -> object: ...

    def get_use_ragged(self) -> bool: ...

    def get_compute_data_type(self) -> thor.DataType: ...

class Convolution2d(TrainableLayer):
    """
    2D convolution layer.

    Builds a trainable 2D convolutional layer with optional activation,
    dropout, and batch normalization. This layer applies a bank of
    learnable filters over the spatial dimensions of the input tensor.

    Parameters
    ----------
    network : thor.Network
        The network that the layer should be added to. The network
        owns the layer and manages its lifetime.
    feature_input : thor.Tensor
        Input feature tensor for this layer.
        Expected layout matches the underlying Thor tensor convention
        of CHW on the API side. The physical implementation side adds
        the batch layer and uses NCHW.
    num_output_channels : int
        Number of output channels produced by the layer.
    filter_height : int
        Height of each convolution filter (kernel size in the vertical
        dimension).
    filter_width : int
        Width of each convolution filter (kernel size in the horizontal
        dimension).
    vertical_stride : int, default 1
        Stride of the convolution in the vertical direction.
    horizontal_stride : int, default 1
        Stride of the convolution in the horizontal direction.
    padding : {"valid", "same"} or tuple[int, int, int, int], default "valid"
        Padding policy. ``"valid"`` applies no padding. ``"same"`` uses
        SAME_UPPER semantics, choosing output spatial dimensions as
        ``ceil(input / stride)`` and assigning any odd extra padding to
        the bottom/right sides. A length-4 tuple specifies explicit
        ``(top, bottom, left, right)`` zero-padding.
    has_bias : bool, default True
        Whether to learn an additive bias per output channel.
    activation : thor.Activation or None, default thor.activations.Gelu()
        Activation to apply after the convolution
        Pass ``None`` to not use any activation and keep the layer
        purely linear.
    weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
        Initializer for the convolution kernel weights.
    biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
        Initializer for the bias vector.
    epilogue : thor.physical.Expression or None, default None
        Optional expression applied after convolution, bias, and activation.
        Build it from ``Convolution2d.epilogue_input()`` and, when needed,
        ``Convolution2d.epilogue_aux_input(name)``.
    epilogue_inputs : dict[str, thor.Tensor] or None, default None
        Named auxiliary input tensors consumed by the epilogue expression.
        This is intended for residual-style epilogues such as
        ``relu(Convolution2d.epilogue_input() + Convolution2d.epilogue_aux_input("residual"))``.
    dilation : int or tuple[int, int], default 1
        Spacing between kernel elements. An integer applies the same dilation
        vertically and horizontally; a length-2 sequence specifies
        ``(vertical_dilation, horizontal_dilation)``.
    compute_data_type : thor.DataType, default thor.DataType.fp32
        ``fp32`` requests strict FP32 convolution math. ``tf32`` explicitly
        permits TensorFloat-32 execution for FP32 input, weight, and output storage.
    """

    def __init__(self, network: thor.Network, feature_input: thor.Tensor, num_output_channels: int, filter_height: int, filter_width: int, vertical_stride: int = 1, horizontal_stride: int = 1, padding: object = 'valid', has_bias: bool = True, activation: object | None = '__thor_default_activation__', weights_initializer: thor.initializers.Initializer | None = None, biases_initializer: thor.initializers.Initializer | None = None, epilogue: object | None = None, epilogue_inputs: object | None = None, dilation: object = 1, groups: int = 1, compute_data_type: thor.DataType = thor.DataType.fp32) -> None: ...

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the primary tensor input expression expected by a Convolution2d epilogue.
        """

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return a named auxiliary tensor input expression for a Convolution2d epilogue.
        Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
        """

    def get_feature_output(self) -> thor.Tensor:
        """
        Return the output tensor produced by this layer.

        Returns
        -------
        thor.Tensor
            The feature output tensor handle.
        """

    def get_compute_data_type(self) -> thor.DataType: ...

class Convolution3d(TrainableLayer):
    """
    3D convolution layer.

    Builds a trainable 3D convolutional layer with optional activation.
    Omitted activation defaults to ``thor.activations.Gelu()``; pass
    ``None`` to keep the layer linear.
    The API tensor layout is CDHW; the physical implementation adds the
    batch dimension and uses NCDHW. ``groups`` partitions input and output
    channels using standard grouped-convolution semantics. Activations are stitched into the
    expression before the implementation CustomLayer is constructed.
    ``epilogue`` may be a ``thor.physical.Expression`` built from
    ``Convolution3d.epilogue_input()`` and is applied after activation.
    ``compute_data_type=thor.DataType.fp32`` requests strict FP32 convolution
    math. ``thor.DataType.tf32`` explicitly permits TensorFloat-32 execution
    for FP32 input, weight, and output storage.
    """

    def __init__(self, network: thor.Network, feature_input: thor.Tensor, num_output_channels: int, filter_depth: int, filter_height: int, filter_width: int, depth_stride: int = 1, vertical_stride: int = 1, horizontal_stride: int = 1, depth_padding: int = 0, vertical_padding: int = 0, horizontal_padding: int = 0, has_bias: bool = True, activation: object | None = '__thor_default_activation__', weights_initializer: thor.initializers.Initializer | None = None, biases_initializer: thor.initializers.Initializer | None = None, epilogue: object | None = None, epilogue_inputs: object | None = None, groups: int = 1, compute_data_type: thor.DataType = thor.DataType.fp32) -> None: ...

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the single tensor input expression expected by a Convolution3d epilogue.
        """

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return a named auxiliary tensor input expression for a Convolution3d epilogue.
        Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
        """

    def get_feature_output(self) -> thor.Tensor:
        """Return the output tensor produced by this layer."""

    def get_compute_data_type(self) -> thor.DataType: ...

class Flatten(Layer):
    def __init__(self, network: thor.Network, feature_input: object, num_output_dimensions: int) -> None:
        """
        Create and attach a Flatten layer to a Network.

        For ragged inputs, num_output_dimensions applies only to the trailing
        per-token value shape. The packed row axis and offsets are preserved.
        """

    def get_feature_output(self) -> object:
        """Return the output tensor produced by this layer."""

class FiniteCheck(Layer):
    def __init__(self, network: thor.Network, feature_input: object, tensor_label: str = '', check_forward: bool = True, check_backward: bool = True, fail_on_non_finite: bool = True, max_reported_indices: int = 8, enabled: bool = True) -> None:
        """
        Create and attach a zero-copy finite-value diagnostic layer.

        Set ``enabled=False`` to leave the layer in the model as a zero-copy no-op. A
        disabled FiniteCheck allocates no diagnostic workspace, launches no check, and
        does not synchronize execution.

        The forward activation and, when a backward path exists, the incoming gradient
        are checked for NaN and infinity values. The layer aliases its input storage in
        both directions and allocates no feature or gradient tensor of its own. For a
        ``RaggedTensor``, only the authoritative active packed prefix ending at
        ``offsets[B]`` is checked; undefined inactive capacity is deliberately ignored
        and the exact row partition is preserved on the output.

        On a failure, the report includes the user label, direction, tensor role, API
        and physical tensor ids, dtype, shape, checked element count, counts of
        NaN/+Inf/-Inf, and sample flat and multidimensional indices.
        ``fail_on_non_finite=True`` raises immediately; ``False`` writes the report to
        stderr and continues.

        FiniteCheck is intentionally a debugging barrier. GPU checks synchronize the
        layer stream so that a host-visible report or exception is deterministic, and
        therefore should not be left enabled in performance runs. Thor emits a warning
        when an enabled FiniteCheck is first stamped.
        """

    def get_feature_output(self) -> object:
        """
        Return the logical output produced by this layer. Ragged inputs return a thor.RaggedTensor.
        """

    def get_use_ragged(self) -> bool: ...

    def get_tensor_label(self) -> str: ...

    def get_enabled(self) -> bool: ...

    def get_check_forward(self) -> bool: ...

    def get_check_backward(self) -> bool: ...

    def get_fail_on_non_finite(self) -> bool: ...

    def get_max_reported_indices(self) -> int: ...

class Embedding(TrainableLayer):
    """
    Sparse-gradient embedding lookup layer.

    Embedding maps an integer Tensor or RaggedTensor of token ids to a floating output whose
    final dimension is ``embedding_dim``. Its output shape is the input index shape
    with ``embedding_dim`` appended. Ragged inputs preserve the exact input row partition;
    inactive packed index capacity is outside the logical operation and is never read.

    This layer intentionally does not implement dense table gradients. Training uses
    sparse row updates; the first backend slice supports plain SGD for fp32 embedding
    tables, with fp16/bf16/fp32 forward lookup support.
    """

    def __init__(self, network: thor.Network, feature_input: object, vocabulary_size: int, embedding_dim: int, weights_data_type: object | None = None, padding_index: object | None = None, sparse_gradients: bool = True, weights_initializer: thor.initializers.Initializer | None = None, weights_optimizer: thor.optimizers.Optimizer | None = None) -> None: ...

    def get_feature_output(self) -> object:
        """Return the output tensor produced by this layer."""

    @property
    def vocabulary_size(self) -> int: ...

    @property
    def embedding_dim(self) -> int: ...

    @property
    def weights_data_type(self) -> thor.DataType: ...

    @property
    def padding_index(self) -> int | None: ...

    @property
    def sparse_gradients(self) -> bool: ...

    @property
    def use_ragged(self) -> bool: ...

class Einsum(MultiConnectionLayer):
    def __init__(self, network: thor.Network, equation: str, feature_inputs: object) -> None:
        """
        Create and attach a symbolic Einsum layer to a Network.

        The equation describes per-example feature dimensions. Thor's physical batch
        axis is implicit and is always preserved. Einsum owns no trainable parameters;
        all operands are ordinary graph tensors and gradients propagate to every live
        operand occurrence.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        equation : str
            Explicit einsum equation, for example ``"ik,kj->ij"``.
        feature_inputs : sequence[thor.Tensor]
            Operand tensors in equation order. Repeating the same symbolic tensor in
            multiple operand positions is supported.
        """

    def get_equation(self) -> str: ...

    def get_feature_output(self) -> thor.Tensor:
        """Return the symbolic tensor produced by this einsum layer."""

class FullyConnected(TrainableLayer):
    """
    Fully connected layer.

    Computes an affine projection followed by the optional activation. ``output_dropout_probability``
    applies dropout after that branch. When ``residual_input`` is supplied, the exact training contract is

        residual_input + dropout(activation(W x + b))

    with omitted operations removed. When output dropout is inactive, Thor keeps the residual add adjacent
    to the projection so the compiler can use the existing GEMM residual/beta fusion where legal. When
    output dropout is active, Thor uses the shared native fused dropout+residual post-op. Validation and
    inference always use the deterministic no-dropout branch. Ragged residuals must share the exact input
    row partition.

    Parameters
    ----------
    network : thor.Network
        The network that the layer should be added to.
    feature_input : thor.Tensor or thor.RaggedTensor
        Input feature tensor for this layer. Ragged inputs are projected tokenwise over their
        packed values and preserve the row partition.
    num_output_features : int
        Number of output features (units) produced by this layer.
    has_bias : bool, default True
        Whether to learn an additive bias term.
    preserve_prefix_dimensions : bool or None, default None
        When omitted, dense inputs default to False and ragged inputs default to True.
        If False, all non-batch dense input dimensions are flattened into one dense feature vector.
        If True, only the final input dimension is treated as features and preceding logical
        dimensions are preserved in the output. Ragged inputs require prefix preservation.
    activation : thor.Activation or None, default thor.activations.Gelu()
        Activation to apply after the linear transform. Pass ``None`` to keep the layer purely
        linear. Ragged FullyConnected uses the same activation contract as dense FullyConnected;
        the activation must support standalone ragged execution.
    weights_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
        Initializer for the weight matrix.
    biases_initializer : thor.initializers.Initializer, default thor.initializers.Glorot()
        Initializer for the bias vector.
    weights_data_type : thor.DataType or None, default None
        Storage type for the weight matrix. When omitted, uses the feature input type.
    compute_data_type : thor.DataType or None, default None
        Arithmetic type for the matrix multiply. When omitted, uses the feature input type, so
        FP32 inputs default to strict FP32 compute. Pass ``thor.DataType.tf32`` to opt into TF32.
    output_data_type : thor.DataType or None, default None
        Storage type for the layer output. When omitted, uses the feature input type.
    output_dropout_probability : float, default 0.0
        Dropout probability on the final FullyConnected branch after bias and activation.
    output_dropout_seed : int or None, default None
        Optional deterministic Philox seed. When omitted with dropout enabled, Thor chooses an independent
        per-layer seed and persists it in the architecture.
    residual_input : thor.Tensor, thor.RaggedTensor, or None, default None
        Optional skip tensor. The exact contract is ``residual_input + dropout(fc_output)`` during training.
        Ragged residuals must share the exact row partition of ``feature_input``.
    epilogue : thor.physical.Expression or None, default None
        Optional expression applied after the affine transform and activation.
        Build it from ``FullyConnected.epilogue_input()``.
    epilogue_inputs : dict[str, thor.Tensor or thor.RaggedTensor] or None, default None
        Named auxiliary tensors referenced by ``FullyConnected.epilogue_aux_input()``.
        Ragged FullyConnected requires every auxiliary to be a RaggedTensor sharing the exact
        row partition of ``feature_input``.
    """

    def __init__(self, network: thor.Network, feature_input: object, num_output_features: int, has_bias: bool = True, activation: object | None = '__thor_default_activation__', weights_initializer: thor.initializers.Initializer | None = None, biases_initializer: thor.initializers.Initializer | None = None, weights_optimizer: thor.optimizers.Optimizer | None = None, biases_optimizer: thor.optimizers.Optimizer | None = None, epilogue: object | None = None, epilogue_inputs: object | None = None, preserve_prefix_dimensions: bool | None = None, weights_constraints: object | None = None, biases_constraints: object | None = None, weights_data_type: object | None = None, compute_data_type: object | None = None, output_data_type: object | None = None, output_dropout_probability: float = 0.0, output_dropout_seed: int | None = None, residual_input: object | None = None) -> None: ...

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the single tensor input expression expected by a FullyConnected epilogue.
        """

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return a named auxiliary tensor input expression for a FullyConnected epilogue.
        Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
        Ragged FullyConnected requires each auxiliary binding to be a ``thor.RaggedTensor``
        with the exact same row partition as ``feature_input``.
        """

    def get_weights_data_type(self) -> thor.DataType: ...

    def get_compute_data_type(self) -> thor.DataType: ...

    def get_output_data_type(self) -> thor.DataType: ...

    def get_feature_output(self) -> object:
        """
        Return the logical output produced by this layer.

        Returns
        -------
        thor.Tensor or thor.RaggedTensor
            The feature output handle. Ragged inputs produce RaggedTensor outputs with the
            same row partition.
        """

    def get_use_ragged(self) -> bool: ...

    def get_output_dropout_probability(self) -> float: ...

    def get_output_dropout_seed(self) -> int: ...

    def get_use_residual(self) -> bool: ...

    def get_residual_input(self) -> object: ...

    def set_training_dropout_enabled(self, enabled: bool) -> None: ...

    def is_training_dropout_enabled(self) -> bool: ...

class InstanceNorm(TrainableLayer):
    """
    Instance normalization over each sample/channel's contiguous spatial region.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor
        Input feature tensor with API dimensions [C, spatial...]. The runtime batch dimension is added by stamping.
    epsilon : float, default 1e-5
        Positive numerical-stability epsilon.
    parameter_data_type : thor.DataType, default thor.DataType.fp32
        Data type for scale and bias. Thor currently requires fp32 for cuDNN Frontend InstanceNorm.
    """

    def __init__(self, network: thor.Network, feature_input: thor.Tensor, epsilon: float = 1e-05, parameter_data_type: thor.DataType = thor.DataType.fp32, weights_initializer: thor.initializers.Initializer | None = None, biases_initializer: thor.initializers.Initializer | None = None, weights_optimizer: thor.optimizers.Optimizer | None = None, biases_optimizer: thor.optimizers.Optimizer | None = None) -> None: ...

    def get_feature_output(self) -> thor.Tensor:
        """Return the output tensor produced by this layer."""

    def get_channel_count(self) -> int: ...

    def get_epsilon(self) -> float: ...

    def get_parameter_data_type(self) -> thor.DataType: ...

class LayerNorm(TrainableLayer):
    """
    Layer normalization over a contiguous trailing normalized shape.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor or thor.RaggedTensor
        Input feature tensor for this layer. Ragged inputs are normalized token-wise over their single trailing channel dimension.
    normalized_shape : Sequence[int] or None, default None
        Trailing feature dimensions to normalize over.  None normalizes the final feature dimension.
    epsilon : float, default 1e-5
        Positive numerical-stability epsilon.
    parameter_data_type : thor.DataType, default thor.DataType.fp32
        Data type for scale and bias.  Thor currently requires fp32 for cuDNN Frontend LayerNorm.
    """

    def __init__(self, network: thor.Network, feature_input: object, normalized_shape: object | None = None, epsilon: float = 1e-05, parameter_data_type: thor.DataType = thor.DataType.fp32, weights_initializer: thor.initializers.Initializer | None = None, biases_initializer: thor.initializers.Initializer | None = None, weights_optimizer: thor.optimizers.Optimizer | None = None, biases_optimizer: thor.optimizers.Optimizer | None = None) -> None: ...

    def get_feature_output(self) -> object:
        """
        Return the logical output produced by this layer. Ragged inputs preserve their row partition.
        """

    def get_use_ragged(self) -> bool: ...

    def get_normalized_shape(self) -> list[int]: ...

    def get_epsilon(self) -> float: ...

    def get_parameter_data_type(self) -> thor.DataType: ...

class RMSNorm(TrainableLayer):
    """
    Root Mean Square Layer Normalization over a contiguous trailing normalized shape.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor or thor.RaggedTensor
        Input feature tensor for this layer. Ragged inputs are normalized token-wise over their trailing value dimensions.
    normalized_shape : Sequence[int] or None, default None
        Trailing feature dimensions to normalize over. None normalizes the final feature dimension.
    epsilon : float, default 1e-5
        Positive numerical-stability epsilon.
    parameter_data_type : thor.DataType or None, default None
        Data type for scale weights. None chooses fp32.
    epilogue : thor.physical.Expression or None, default None
        Optional expression applied after RMSNorm. Build it from ``RMSNorm.epilogue_input()``.
        A Swish/SiLU epilogue can use the cuDNN Frontend RMSNorm + SiLU inference fusion when the
        feature input, output, and scale weights are bf16.
    epilogue_inputs : dict[str, thor.Tensor | thor.RaggedTensor] or None, default None
        Named auxiliary tensors referenced by ``RMSNorm.epilogue_aux_input()``. With a ragged
        feature input, every auxiliary must be a RaggedTensor sharing its exact row partition.
    """

    def __init__(self, network: thor.Network, feature_input: object, normalized_shape: object | None = None, epsilon: float = 1e-05, parameter_data_type: object | None = None, weights_initializer: thor.initializers.Initializer | None = None, weights_optimizer: thor.optimizers.Optimizer | None = None, epilogue: object | None = None, epilogue_inputs: object | None = None) -> None: ...

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the single tensor input expression expected by an RMSNorm epilogue.
        """

    @staticmethod
    def epilogue_aux_input(name: str, output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return a named auxiliary tensor input expression for an RMSNorm epilogue.
        Bind the same name to a tensor with the ``epilogue_inputs`` constructor argument.
        With a ragged feature input, the binding must be a ``thor.RaggedTensor`` sharing the exact row partition.
        """

    def get_feature_output(self) -> object:
        """
        Return the logical output produced by this layer. Ragged inputs preserve their row partition.
        """

    def get_use_ragged(self) -> bool: ...

    def get_normalized_shape(self) -> list[int]: ...

    def get_epsilon(self) -> float: ...

    def get_parameter_data_type(self) -> thor.DataType: ...

class ScaledDotProductAttention(CustomLayer):
    """
    cuDNN-backed scaled dot-product attention layer for already-projected Q/K/V tensors.

    Dense API tensor shapes omit batch.  ``tensor_layout='bhsd'`` means
    ``[heads, sequence, head_dim]`` and ``tensor_layout='bshd'`` means
    ``[sequence, heads, head_dim]``.  Canonical ragged inputs instead use
    ``thor.RaggedTensor`` packed values shaped ``[max_total_values, heads, head_dim]``
    with a separate logical row partition.  cuDNN-specific lengths and element
    offsets remain private backend metadata.

    FP16/BF16 production support:

    * Q/K/V/O must all use the same FP16 or BF16 dtype.  ``compute_data_type`` should
      be FP32 and ``output_data_type`` should normally match Q/K/V.
    * Forward and backward are supported for self-attention, cross-attention, MHA,
      GQA, and MQA.  Query heads must be an integer multiple of key/value heads.
    * Masks: ``none``, ``causal_top_left``, ``causal_bottom_right``,
      ``sliding_window_top_left``, and ``sliding_window_bottom_right``.
    * ALiBi requires a causal/sliding diagonal mask and ``diagonal_right_bound == 0``.
      Positive right bounds with ALiBi are rejected because cuDNN rejects that graph.
    * ``bias_input`` is score-space additive bias with API shape
      ``[1|Hq, 1|Sq, 1|Skv]`` and dtype equal to ``compute_data_type``.  Forward
      supports sequence broadcast.  Backward materializes sequence-broadcast bias to
      dense score space before cuDNN backward and reduces dBias back to the public
      bias shape.  Ragged + additive-bias backward is rejected.
    * Dense/padded attention may use ``sequence_lengths`` or the explicit
      ``query_sequence_lengths``/``key_value_sequence_lengths`` pair.
    * Query and key/value domains may be dense or ragged independently. Key and value
      must use the same domain; a ragged key/value pair must share one canonical row
      partition. When either domain is ragged, BSHD/token-major layout is required and
      Thor synthesizes a uniform private row partition for the dense side before calling
      cuDNN. The output is ragged exactly when the query is ragged. Supplying dense
      sequence-length metadata together with any RaggedTensor input is rejected.
    * ``dropout_probability``/``dropout_seed``/``dropout_offset`` expose cuDNN Philox
      attention dropout.  Thor advances the runtime dropout offset by
      ``B * Hq * Sq * Skv``.

    Experimental FP8 forward-only support:

    * Enable by passing all eight scalar FP32 tensors:
      ``fp8_descale_q``, ``fp8_descale_k``, ``fp8_descale_v``, ``fp8_descale_s``,
      ``fp8_scale_s``, ``fp8_scale_o``, ``fp8_amax_s``, and ``fp8_amax_o``.
    * Q/K/V/O must all be the same FP8 format, either E4M3 or E5M2.  QK and V head
      dimensions must be multiples of 16 and no larger than 128 on the validated
      production surface.
    * FP8 backward is not supported.  FP8 additive bias, dropout, ALiBi, ragged,
      paged KV, bottom-right/decode masks, sliding-window masks, and decode-style
      ``Sq=1, Skv>1`` are rejected on the validated public surface.
    * FP8 padding masks / sequence lengths are supported for forward; canonical RaggedTensor
      inputs remain disabled for FP8.

    Important combination rules:

    * Bottom-right/decode masks currently require additive bias, ALiBi, and dropout
      to be disabled in the production cuDNN primary SDPA path.
    * Paged KV cache is not exposed by this layer.  It is available only through the
      low-level physical expression API as an inference-only FP16/BF16 path with
      padding-mask sequence lengths, no bias, and no dropout.
    * Experimental cuDNN support-surface probe environment variables can bypass some
      guards for measurement, but those combinations are not user-facing support
      guarantees.
    """

    def __init__(self, network: thor.Network, query_input: object, key_input: object | None = None, value_input: object | None = None, bias_input: thor.Tensor | None = None, tensor_layout: str | None = None, mask_kind: str | None = 'none', diagonal_left_bound: int = 0, diagonal_right_bound: int = 0, use_alibi_mask: bool = False, attention_scale: float | None = None, compute_data_type: thor.DataType | None = thor.DataType.fp32, output_data_type: thor.DataType | None = None, sequence_lengths: thor.Tensor | None = None, query_sequence_lengths: thor.Tensor | None = None, key_value_sequence_lengths: thor.Tensor | None = None, dropout_probability: float | None = 0.0, dropout_seed: int = 0, dropout_offset: int = 0, fp8_descale_q: thor.Tensor | None = None, fp8_descale_k: thor.Tensor | None = None, fp8_descale_v: thor.Tensor | None = None, fp8_descale_s: thor.Tensor | None = None, fp8_scale_s: thor.Tensor | None = None, fp8_scale_o: thor.Tensor | None = None, fp8_amax_s: thor.Tensor | None = None, fp8_amax_o: thor.Tensor | None = None) -> None: ...

    def get_feature_output(self) -> object:
        """Return the logical Tensor or RaggedTensor produced by this layer."""

    def get_tensor_layout(self) -> str: ...

    def get_mask_kind(self) -> str: ...

    def get_diagonal_left_bound(self) -> int: ...

    def get_diagonal_right_bound(self) -> int: ...

    def get_use_alibi_mask(self) -> bool: ...

    def get_attention_scale(self) -> float | None: ...

    def get_dropout_probability(self) -> float: ...

    def get_dropout_seed(self) -> int: ...

    def get_dropout_offset(self) -> int: ...

    def get_use_sequence_lengths(self) -> bool: ...

    def get_use_ragged_input(self) -> bool: ...

    def get_query_is_ragged(self) -> bool: ...

    def get_key_value_is_ragged(self) -> bool: ...

    def get_use_bias(self) -> bool: ...

    def get_bias_input(self) -> thor.Tensor | None: ...

    def get_query_sequence_lengths_input(self) -> thor.Tensor | None: ...

    def get_key_value_sequence_lengths_input(self) -> thor.Tensor | None: ...

    def get_query_ragged_input(self) -> thor.RaggedTensor | None: ...

    def get_key_ragged_input(self) -> thor.RaggedTensor | None: ...

    def get_value_ragged_input(self) -> thor.RaggedTensor | None: ...

    def get_use_fp8_forward_scaling(self) -> bool: ...

    def get_fp8_descale_q_input(self) -> thor.Tensor | None: ...

    def get_fp8_descale_k_input(self) -> thor.Tensor | None: ...

    def get_fp8_descale_v_input(self) -> thor.Tensor | None: ...

    def get_fp8_descale_s_input(self) -> thor.Tensor | None: ...

    def get_fp8_scale_s_input(self) -> thor.Tensor | None: ...

    def get_fp8_scale_o_input(self) -> thor.Tensor | None: ...

    def get_fp8_amax_s_input(self) -> thor.Tensor | None: ...

    def get_fp8_amax_o_input(self) -> thor.Tensor | None: ...

    def get_compute_data_type(self) -> thor.DataType: ...

    def get_output_data_type(self) -> thor.DataType: ...

class ScaleGradient(Layer):
    def __init__(self, network: thor.Network, feature_input: object, scale: float) -> None:
        """
        Create and attach a ScaleGradient layer to a Network.

        Forward is an identity alias of ``feature_input``. Ragged inputs preserve their exact row partition. During backward propagation,
        the gradient passed upstream is multiplied by ``scale``. The downstream branch
        and any trainable layers after ScaleGradient receive their ordinary gradients.

        ``scale=0`` blocks the numerical gradient while preserving a backward tensor
        path. Negative scales are allowed and can be used for gradient reversal.
        """

    def get_feature_output(self) -> object:
        """
        Return the logical output produced by this layer. Ragged inputs return a thor.RaggedTensor.
        """

    def get_use_ragged(self) -> bool: ...

    def get_scale(self) -> float:
        """Return the backward gradient scale."""

class SegmentedBroadcast(MultiConnectionLayer):
    """
    Broadcast one dense value per batch row to every active token in a ragged row.

    ``feature_input`` is a normal dense per-example tensor. ``partition_input``
    provides only the canonical row offsets/capacity; its packed values are not read.
    The output is a ``thor.RaggedTensor`` with the exact same offsets object and with
    trailing value dimensions equal to ``feature_input``. Values must be FP16, BF16,
    or FP32. FP64 is intentionally unsupported.
    """

    def __init__(self, network: thor.Network, feature_input: thor.Tensor, partition_input: thor.RaggedTensor) -> None: ...

    def get_feature_output(self) -> thor.RaggedTensor: ...

    def get_partition_input(self) -> thor.RaggedTensor: ...

class SegmentedLogSoftmax(MultiConnectionLayer):
    """
    Log-softmax across the active tokens of each ragged row.

    Each trailing component is normalized independently over its row's variable-length
    token axis. The exact canonical offsets object is preserved and inactive packed
    capacity is excluded. Values must be FP16, BF16, or FP32; FP64 is intentionally
    unsupported.
    """

    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor) -> None: ...

    def get_feature_output(self) -> thor.RaggedTensor: ...

class SegmentedReduction(MultiConnectionLayer):
    """
    Reduce each row of a packed ``thor.RaggedTensor`` independently.

    The row partition supplies the reduction domains. ``sum``, ``mean``, ``min``,
    and ``max`` preserve every fixed trailing value dimension and return a normal dense
    ``thor.Tensor`` feature shape. At execution time the physical
    shape is ``[batch_size, *trailing_dimensions]`` (or ``[batch_size, 1]`` for
    scalar ragged values). Empty rows follow the existing Thor segmented-reduction semantics.
    """

    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor, reduction_type: SegmentedReduction.Type) -> None: ...

    class Type(enum.Enum):
        sum = 0

        mean = 1

        min = 2

        max = 3

    def get_feature_output(self) -> thor.Tensor: ...

    @property
    def reduction_type(self) -> SegmentedReduction.Type: ...

class SegmentedSoftmax(MultiConnectionLayer):
    """
    Softmax across the active tokens of each ragged row.

    Each trailing component is normalized independently over its row's variable-length
    token axis. The exact canonical offsets object is preserved and inactive packed
    capacity is excluded. Values must be FP16, BF16, or FP32; FP64 is intentionally
    unsupported. This is distinct from ordinary ``Softmax``, which is not a segmented
    sequence-axis operation.
    """

    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor) -> None: ...

    def get_feature_output(self) -> thor.RaggedTensor: ...

class Slice(Layer):
    """
    Slice a contiguous window from one logical tensor axis.

    ``feature_input`` may be a dense ``thor.Tensor`` or packed ``thor.RaggedTensor``. For ragged input, ``axis`` addresses only the fixed trailing value dimensions and the row partition is preserved.

    The batch dimension is excluded from ``axis``. Negative ``start`` values are
    resolved relative to the end of the selected logical axis. The operation is
    serialized declaratively as ``axis``, ``start``, and ``length`` and remains
    batch-polymorphic when the network is placed, cloned into training phases, or
    saved and reloaded.
    """

    def __init__(self, network: thor.Network, feature_input: object, axis: int, start: int, length: int) -> None: ...

    def get_feature_output(self) -> object: ...

    def get_use_ragged(self) -> bool: ...

    @property
    def axis(self) -> int: ...

    @property
    def start(self) -> int: ...

    @property
    def length(self) -> int: ...

class StopGradient(Layer):
    def __init__(self, network: thor.Network, feature_input: object) -> None:
        """
        Create and attach a StopGradient layer to a Network.

        Forward is an identity alias of ``feature_input``. Backward does not propagate
        an error tensor through this layer, making the gradient barrier explicit in the
        network graph. Ragged inputs preserve their exact row partition.
        """

    def get_feature_output(self) -> object:
        """
        Return the logical output produced by this layer. Ragged inputs return a thor.RaggedTensor.
        """

    def get_use_ragged(self) -> bool: ...

class NetworkInput(Layer):
    """
    Create and attach a NetworkInput to send data into a Network.

    Parameters
    ----------
    network : thor.Network
        The network that the layer should be added to.
    name : str
        Name of this network input.
    dimensions : list[int]
        Dimension sizes for the input tensor **excluding** the batch dimension.
        The batch dimension is added later when compiling the network.
        Note: the batch dimension is never specified in API layer tensors,
              the batch dimension is only added when stamping down a physical network instance.
    data_type : thor.DataType
        Data type of the input tensor (e.g. thor.DataType.fp16).
    dimensions_include_batch : bool, default False
        When True, ``dimensions`` already includes the batch dimension.
        This is primarily for internal network-composition runtimes.
    pass_through_source : thor.Tensor | None, default None
        Internal network-composition hook.  When supplied, this NetworkInput
        is an API-level pass-through alias of the source tensor.  It is not
        stamped as an external network input and does not allocate or copy
        through an input staging tensor.
    external : bool, default True
        Whether this input may be satisfied by external data when no active
        phase NetworkOutput with the same name is available.  Pass-through
        inputs are always treated as non-external.
    """

    def __init__(self, network: thor.Network, name: str, dimensions: Sequence[int], data_type: thor.DataType, dimensions_include_batch: bool = False, pass_through_source: thor.Tensor | None = None, external: bool | None = True) -> None: ...

    def get_feature_output(self) -> thor.Tensor:
        """
        Return the output tensor produced by this layer.

        Returns
        -------
        thor.Tensor
            The feature output tensor handle.
        """

    def is_external(self) -> bool: ...

    def version(self) -> str: ...

class RaggedFilter(MultiConnectionLayer):
    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor, mask_input: thor.RaggedTensor) -> None:
        """
        Stable-filter every row of a rank-1 RaggedTensor with one BOOLEAN predicate per token.

        ``mask_input`` must be a scalar BOOLEAN RaggedTensor sharing the exact same
        canonical offsets tensor and row-partition descriptor as ``feature_input``.
        Selected active tokens preserve their row-local order and are compacted into a
        new packed values tensor with a newly produced canonical offsets tensor. Neither
        forward nor backward reads inactive packed capacity. Backward writes zero to
        active filtered-out feature positions and scatters gradients only to retained
        positions; the BOOLEAN mask is non-differentiable.
        """

    def get_feature_output(self) -> thor.RaggedTensor: ...

    def get_feature_input(self) -> thor.RaggedTensor: ...

    def get_mask_input(self) -> thor.RaggedTensor: ...

class RaggedGather(MultiConnectionLayer):
    def __init__(self, network: thor.Network, source_input: thor.RaggedTensor, indices_input: thor.RaggedTensor) -> None:
        """
        Gather tokens independently within every row of a rank-1 RaggedTensor.

        ``indices_input`` must contain scalar UINT32 or UINT64 row-local indices. Its
        row partition Q defines the output partition exactly, while ``source_input``
        provides source partition P and the output value dtype/trailing shape. Thus
        source and indices may have different row lengths. Duplicate indices are valid
        and preserve their occurrence order; backward sums their gradient contributions
        into the selected source token. Inactive packed source/index capacity is never
        read.
        """

    def get_feature_output(self) -> thor.RaggedTensor: ...

    def get_source_input(self) -> thor.RaggedTensor: ...

    def get_indices_input(self) -> thor.RaggedTensor: ...

class RaggedRowLengths(Layer):
    """
    Materialize canonical ragged row lengths as dense INT32 logical ``[1]`` values.

    For offsets ``[0, 371, 558, 612]`` the physical output is ``[[371], [187], [54]]``.
    The layer depends only on row-partition offsets, not on packed values.
    """

    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor) -> None: ...

    def get_feature_output(self) -> thor.Tensor: ...

class RaggedToPaddedDense(MultiConnectionLayer):
    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor, padding_value: float = 0.0) -> None:
        """
        Convert a rank-1 RaggedTensor to a normal padded dense tensor.

        ``feature_input`` must declare ``max_values_per_row``. The logical output shape
        is ``[max_values_per_row, *trailing]`` and the physical stamped shape is
        ``[B, max_values_per_row, *trailing]``. Active tokens are copied row-by-row and
        all remaining padded positions are filled with ``padding_value``. Backward
        ignores gradients in padded positions and returns only active packed gradients.
        """

    def get_feature_output(self) -> thor.Tensor: ...

    def get_feature_input(self) -> thor.RaggedTensor: ...

    @property
    def padding_value(self) -> float: ...

class RaggedSequenceConcatenate(MultiConnectionLayer):
    def __init__(self, network: thor.Network, feature_inputs: object) -> None:
        """
        Concatenate rank-1 ragged inputs along their variable-length sequence axis.

        Every input must have the same logical batch size, values dtype, offsets dtype,
        and trailing value shape. Row partitions may differ. For each logical row, the
        output contains row 0 from every input in argument order, then row 1 from every
        input, and so on. The layer explicitly produces a new canonical offsets tensor;
        it does not reuse any input partition. Inactive packed capacity is never read.
        """

    def get_feature_output(self) -> thor.RaggedTensor: ...

class RaggedSequenceSlice(MultiConnectionLayer):
    def __init__(self, network: thor.Network, feature_input: thor.RaggedTensor, start: int, length: int) -> None:
        """
        Slice every logical row of a rank-1 RaggedTensor along the variable-length sequence axis.

        ``start`` is a non-negative row-local token offset and ``length`` must be
        positive. Each row contributes at most ``length`` tokens beginning at ``start``;
        short rows are clipped independently and rows no longer than ``start`` become
        empty. Selected values are compacted and the layer explicitly produces a new
        canonical offsets tensor rather than preserving the input partition. Inactive
        packed capacity is never read, and backward writes exact zero to active input
        positions outside the selected window while leaving inactive gradient capacity
        undefined.
        """

    def get_feature_output(self) -> thor.RaggedTensor: ...

    @property
    def start(self) -> int: ...

    @property
    def length(self) -> int: ...

class NetworkOutput(Layer):
    """
    Create and attach a NetworkOutput to send data out of a Network.

    Parameters
    ----------
    network : thor.Network
        The network that the layer should be added to.
    name : str
        Name of this network output.
    input_tensor : thor.Tensor
        The tensor whose data the network output will send out of the network.
    data_type : thor.DataType
        Data type of the output tensor (e.g. thor.DataType.fp16).
    external : bool, default True
        Whether this output should be exposed/materialized outside of a
        composed local phase graph. Non-external outputs may still satisfy
        downstream phase NetworkInputs by name.
    """

    def __init__(self, network: thor.Network, name: str, input_tensor: thor.Tensor, data_type: thor.DataType, external: bool = True) -> None: ...

    def get_feature_output(self) -> thor.Tensor:
        """
        Return the output tensor produced by this layer.

        Returns
        -------
        thor.Tensor
            The feature output tensor handle.
        """

    def is_external(self) -> bool: ...

    def version(self) -> str: ...

class RaggedNetworkOutput:
    def __init__(self, network: thor.Network, name: str, input_tensor: thor.RaggedTensor) -> None:
        """
        Expose one logical ragged result from a Network.

        The packed values and row-partition offsets are materialized internally as a
        paired output, but inference returns one
        ``thor.physical.PhysicalRaggedTensor`` under ``name`` rather than exposing the
        component output names.
        """

    def get_name(self) -> str: ...

    def get_input(self) -> thor.RaggedTensor: ...

    def get_feature_output(self) -> thor.RaggedTensor: ...

class Pooling(Layer):
    def __init__(self, network: thor.Network, feature_input: thor.Tensor, type: Pooling.Type, window_height: int, window_width: int, vertical_stride: int = 1, horizontal_stride: int = 1, vertical_padding: int = 0, horizontal_padding: int = 0) -> None:
        """
        Pooling layer that downsamples its input by applying a pooling operation
        (e.g. max or average) over sliding windows.

        This layer supports different pooling types via :class:`thor.Pooling.Type`
        (such as ``Pooling.Type.MAX`` or ``Pooling.Type.AVERAGE``), and allows
        explicit control over window size, stride, and padding in both the
        vertical and horizontal directions.

        Parameters
        ----------
        network : thor.Network
            The network to add this layer into.
        type : thor.Pooling.Type
            The pooling mode to use (e.g. ``Pooling.Type.MAX`` or
            ``Pooling.Type.AVERAGE``).
        window_height : int
            Height of the pooling window (in cells).
        window_width : int
            Width of the pooling window (in cells).
        vertical_stride : int, optional
            Vertical stride of the pooling window. Defaults to 1.
        horizontal_stride : int, optional
            Horizontal stride of the pooling window. Defaults to 1.
        vertical_padding : int, optional
            Amount of zero-padding added to the top and bottom of the input.
            This amount of padding is added to both the top and bottom of the input,
            so vertical_padding=2 creates 4 rows of padding total.
            Defaults to 0.
        horizontal_padding : int, optional
            Amount of zero-padding added to the left and right of the input.
            This amount of padding is added to both the left and right of the input,
            so horizontal_padding=2 creates 4 columns of padding total.
            Defaults to 0.

        Notes
        -----
        The supported tensor layout is NCHW.
        """

    class Type(enum.Enum):
        average = 3

        max = 4

    def get_feature_output(self) -> thor.Tensor:
        """Return the output tensor produced by this layer."""

    def get_output_dimensions(self) -> list[int]: ...

    def get_pooling_type(self) -> Pooling.Type: ...

    def get_window_height(self) -> int: ...

    def get_window_width(self) -> int: ...

    def get_vertical_stride(self) -> int: ...

    def get_horizontal_stride(self) -> int: ...

    def get_vertical_padding(self) -> int: ...

    def get_horizontal_padding(self) -> int: ...

class PaddedDenseToRagged(MultiConnectionLayer):
    def __init__(self, network: thor.Network, feature_input: thor.Tensor, partition_input: thor.RaggedTensor) -> None:
        """
        Pack a normal padded dense tensor into canonical ragged storage.

        ``partition_input`` is the sole source of row membership; only its offsets are
        consumed and the output reuses that exact partition. The dense input logical
        shape is ``[padded_width, *trailing]`` with ``padded_width`` at least
        ``partition_input.max_values_per_row``. Padding cells are ignored. Backward
        materializes dense gradients with exact zeros in padded positions.
        """

    def get_feature_output(self) -> thor.RaggedTensor: ...

    def get_feature_input(self) -> thor.Tensor: ...

    def get_partition_input(self) -> thor.RaggedTensor: ...

class Reshape(Layer):
    """
    Create and attach a Reshape layer to a Network.

    Dense inputs reshape the complete feature tensor. Ragged inputs reshape
    only the trailing dimensions of each packed value; the packed row axis
    and canonical offsets are preserved exactly.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor or thor.RaggedTensor
        Input feature tensor for this layer.
    new_dimensions : list[int]
        Dense output feature shape, or for ragged input the new per-token
        trailing shape. Element count must be preserved.
    """

    def __init__(self, network: thor.Network, feature_input: object, new_dimensions: Sequence[int]) -> None: ...

    def get_feature_output(self) -> object:
        """
        Return the output tensor produced by this layer.

        Returns
        -------
        thor.Tensor or thor.RaggedTensor
            Ragged inputs preserve their exact row partition and reshape only
            the trailing per-token value dimensions.
        """

class Stub(Layer):
    def __init__(self, network: thor.Network, input_tensor: thor.Tensor) -> None:
        """
        Create and attach a Stub layer to a Network.
        When there is a dangling tensor in the execution graph (i.e. it is not
        connected to the input of anything else in the network) then the graph
        compiler will complain about a dangling tensor and abort. If you want
        to graph to compile you can tell network that you are aware and ok with
        the dangling tensor by attaching it as the input to a Stub layer, and
        then the network will compile.

        Parameters
        ----------
        network : thor.Network
            Network the layer should be added to.
        input_tensor : thor.Tensor
            Input feature tensor for this layer.
        """

class Transpose(Layer):
    """
    Create and attach a Transpose layer to a Network.

    The layer swaps the last two feature dimensions. The network batch
    dimension is preserved by the underlying physical expression, so a
    feature tensor shaped [X, Y] is materialized as [Y, X], while the
    stamped physical tensor behaves as [B, X, Y] -> [B, Y, X]. The
    optional output_dtype casts the transposed value before the optional
    epilogue expression is applied.

    Parameters
    ----------
    network : thor.Network
        Network the layer should be added to.
    feature_input : thor.Tensor or thor.RaggedTensor
        Dense inputs must have rank >= 2. Ragged inputs must have at least
        two trailing value dimensions; only those final two dimensions are
        transposed and the row partition is preserved.
    output_dtype : thor.DataType | None, default None
        Optional dtype for the transposed layer output. Defaults to the
        input feature dtype.
    epilogue : thor.physical.Expression | None, default None
        Optional expression applied after the transpose/output dtype cast.
    """

    def __init__(self, network: thor.Network, feature_input: object, output_dtype: object | None = None, epilogue: object | None = None) -> None: ...

    @staticmethod
    def epilogue_input(output_dtype: object | None = None, compute_dtype: object | None = None) -> thor.physical.Expression:
        """
        Return the single tensor input expression expected by a Transpose epilogue.
        """

    def get_feature_output(self) -> object:
        """
        Return the output tensor produced by this layer.

        Returns
        -------
        thor.Tensor or thor.RaggedTensor
            The feature output tensor handle.
        """

    def get_output_data_type(self) -> thor.DataType: ...

class CudaKernelLayer(CustomLayer):
    """
    Convenience wrapper for using a CudaKernelExpression as a CustomLayer.

        This class intentionally does not create a separate serialized layer kind.
        It lowers directly to CustomLayer with ``kernel.as_dynamic_expression()``, so
        the existing CUDA-kernel source inspection and save/load key policy continues
        to apply without a second security path. For training, attach an explicit
        backward CUDA kernel to each differentiable forward output with
        ``CudaKernelExpressionBuilder.backward(...)``; ordinary CustomLayer expressions
        continue to use Thor's automatic differentiation.
    """
