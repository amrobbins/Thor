"""Runtime execution namespace."""

from collections.abc import Sequence
import enum

import thor
import thor.parameters


class PlacedNetwork:
    def save(self, directory: str, overwrite: bool = False, save_optimizer_state: bool = False) -> None: ...

    def get_num_stamps(self) -> int: ...

    def set_training_dropout_enabled(self, enabled: bool) -> None:
        """
        Drain work already submitted by this placement, then enable or disable training-time dropout
        for all controllable physical layers. Configured dropout probabilities are unchanged. Callers
        must not submit batches concurrently with this operation.
        """

    def is_training_dropout_enabled(self) -> bool: ...

    def get_num_training_dropout_controllable_layers(self) -> int: ...

    def infer(self, batch_inputs: dict, stamp_index: int = 0) -> dict:
        """
        Run one inference batch through this placed network stamp.

        Parameters
        ----------
        batch_inputs : dict[str, thor.physical.PhysicalTensor | thor.physical.PhysicalRaggedTensor]
            Logical dense or ragged input fields keyed by NetworkInput/RaggedNetworkInput name.
        stamp_index : int, default 0
            Stamped network instance to execute.

        Returns
        -------
        dict[str, thor.physical.PhysicalTensor | thor.physical.PhysicalRaggedTensor]
            Logical external outputs keyed by NetworkOutput/RaggedNetworkOutput name. Ragged component tensors remain implementation details.
        """

    def get_stamped_network(self, i: int) -> "ThorImplementation::StampedNetwork": ...

    def get_network_name(self) -> str: ...

    def get_num_trainable_layers(self) -> int: ...

    def resolve_parameter_reference(self, parameter_reference: thor.parameters.ParameterReference) -> thor.parameters.BoundParameter: ...

    def resolve_parameter_references(self, parameter_references: Sequence[thor.parameters.ParameterReference]) -> list[thor.parameters.BoundParameter]: ...

    def has_api_tensor(self, tensor: thor.Tensor) -> bool: ...

    def resolve_api_tensor(self, tensor: thor.Tensor) -> thor.Tensor: ...

    def resolve_api_tensors(self, tensors: Sequence[thor.Tensor]) -> list[thor.Tensor]: ...

    def has_network_input(self, name: str) -> bool: ...

    def get_network_input_names(self, stamp_index: int = 0) -> list[str]: ...

class StatusCode(enum.Enum):
    success = 0

    floating_input = 1

    dangling_output = 2

    gpu_out_of_memory = 3

    duplicate_named_network_input = 4

    duplicate_named_network_output = 5

    deadlock_cycle = 6

__all__: list = ['PlacedNetwork', 'StatusCode']

def __dir__() -> list[str]: ...
