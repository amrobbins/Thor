# Historical AlexNet reference

The known-good visual reference is the legacy `AlexNetDemo` Nsight timeline from
an NVIDIA TITAN RTX.  Keep the recovered screenshot in this directory as:

```text
legacy_local_executor_titan_rtx.png
```

Observed properties of that trace:

- dense, nearly continuous kernel execution within and across batches;
- useful overlap across multiple CUDA streams;
- periodic HtoD for the following batch while current-batch compute is active;
- small DtoH transfers off the critical compute path;
- no large host-created gap between consecutive batches.

Known provenance:

```text
executable: AlexNetDemo
GPU:        NVIDIA TITAN RTX
runtime:    legacy LocalExecutor-era Thor
```

Do not infer an exact Thor commit, CUDA version, Nsight version, or batch size
from the screenshot unless those details are recovered independently.
