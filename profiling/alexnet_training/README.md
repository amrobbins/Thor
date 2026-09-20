# AlexNet training timeline profile

This is Thor's first end-to-end runtime scheduling profiling case.  Its purpose
is to make regressions in queue-ahead submission, stream overlap, NetworkInput
buffering, NetworkOutput completion, and batch-to-batch flow visually obvious in
Nsight Systems.

The case runs ordinary FP16 AlexNet through the current `Trainer` /
`NativeQueuedTrainingRunner`.  There is intentionally no legacy `LocalExecutor`
mode: the profiling case is the stable workload we use while unwinding runtime
regressions in the implementation that Thor actually supports today.

## Data

By default the first run streams prefixes of the public
`clane9/imagenet-100` dataset from Hugging Face, preprocesses them to normalized
FP16 CHW 224x224 tensors, and caches the result under:

```text
profiling/data/alexnet_imagenet100/
```

Only the requested subsets are materialized. The default is 20,000 images from
the dataset's `train` split plus all 5,000 images from its `validation` split.
Repeated profiles reuse the preprocessed cache. This gives the profiling case a
real validation phase rather than carving validation examples out of training
data.

The optional Python dependencies are:

```bash
./.venv/bin/pip install datasets Pillow
```

The entire `profiling/data/` tree is ignored by Git. The profiler creates it
automatically, reuses a matching cached subset on later runs, and downloads /
preprocesses the requested subset again when the cache is absent. Use
`--rebuild-data` to force the cached subset to be discarded and rebuilt.
`--cache-dir` can still override the location for one-off experiments.

## Capture

From the repository root:

```bash
./profiling/alexnet_training/profile.sh
```

The script profiles the **current source-tree build**, not whatever Thor wheels
happen to be installed in the venv. It prepends `build/bindings/python` to
`PYTHONPATH`, enables Thor's source-tree CUDA bootstrap mode (the same import
mode used by `check-python`), verifies that `thor` resolves from that build tree,
and only then starts Nsight. This prevents a freshly built `_thor` extension from
being mixed with a stale installed kernel-wheel `libThor.so`.

The launcher implementation is also taken from the current checkout first, from
`bindings/python/src/thor_nsys_profile`, so capture/finalization behavior matches
the source being profiled. `THOR_NSYS_PROFILE` remains an explicit override. If
your CMake build directory is not `build/`, set `THOR_PYTHON_BUILD_DIR` to its
`bindings/python` directory.

If the build-tree extension is missing or stale, rebuild it before profiling:

```bash
cmake --build build --target _thor -j
```

By default it runs one warmup epoch, then captures one steady-state epoch to:

```text
build/profiling/alexnet_training/alexnet_trainer.nsys-rep
```

The default workload is deliberately close to the historical AlexNet workload:

```text
batch size:             512
train examples:        20000 (39 full batches plus the final partial batch)
validation examples:    5000 (dataset validation split)
max in-flight batches:     4
warmup epochs:             1
captured epochs:           1
training data residency: host (device_storage=off)
```

Host residency is intentional.  A core invariant of this case is that HtoD for
batch N+1 should overlap compute for batch N.

Each captured epoch now contains both training and validation. The training
portion remains the primary steady-state scheduling reference; the validation
portion also gives us a repeatable train-to-validation phase transition to
inspect.

Useful overrides:

```bash
./profiling/alexnet_training/profile.sh \
  --batch-size 256 \
  --train-examples 20000 \
  --validate-examples 5000 \
  --max-in-flight-batches 8 \
  --overwrite-profile
```

To run the identical workload without invoking Nsight while still using the
current build-tree bindings:

```bash
THOR_CUDA_BOOTSTRAP_SOURCE_TREE=1 \
PYTHONPATH=build/bindings/python \
./.venv/bin/python3 profiling/alexnet_training/alexnet_training_profile.py \
  --no-nsight
```

For a very short diagnostic loop, cap training batches without changing the
cached dataset:

```bash
./profiling/alexnet_training/profile.sh \
  --max-training-batches-per-epoch 4 \
  --overwrite-profile
```

## Timeline invariants

The historical known-good AlexNet profile is our behavioral reference.  A
healthy current trace should have these properties:

- kernels within a batch are densely submitted rather than separated by large
  host-created idle gaps;
- independent CUDA streams overlap where graph dependencies allow it;
- HtoD for the next batch overlaps current-batch compute;
- DtoH/output handling does not create a batch barrier;
- configured in-flight depth is actually used;
- the next batch's work is already queued before the current batch completes.

When a large GPU idle interval appears, inspect the CUDA API and stream lanes at
that exact boundary and classify it before changing code:

1. next kernels were never submitted -> host scheduling/submission problem;
2. next kernels were submitted but wait on an event -> dependency/synchronization problem;
3. HtoD starts only after compute ends -> input buffering regression;
4. slot reuse waits for output/stat completion -> completion frontier regression.

Fix the first structural divergence from the known-good timeline before
optimizing individual kernel launch costs.
