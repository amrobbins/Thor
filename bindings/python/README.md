# thor-cuda

Official CUDA backend package for the [Thor deep learning framework](https://github.com/amrobbins/Thor).

Thor is a C++/CUDA deep learning framework focused on high-performance tensor expression compilation, fusion, and GPU execution.

## Status

This package is **pre-alpha**.

## Project links

- Repository: https://github.com/amrobbins/Thor
- Issues: https://github.com/amrobbins/Thor/issues

## License

Apache License 2.0. See `LICENSE`.

## Release wheel flow

The public `thor-cuda` wheel build is the complete native release build. From
`bindings/python`, one command compiles the shared host implementation once,
compiles each supported SM once, and emits the three raw wheels together:

```bash
rm -rf build dist wheelhouse
python -m build --wheel
```

The expected raw release set is:

```text
thor_cuda-<version>-cp312-cp312-linux_x86_64.whl
thor_cuda_kernels_sm89-<version>-py3-none-linux_x86_64.whl
thor_cuda_kernels_sm120-<version>-py3-none-linux_x86_64.whl
```

In the `manylinux_2_28` release container, repair and verify the complete set
with the source-tree CMake release entry point:

```bash
cd ../..
cmake -DTHOR_RELEASE_ACTION=repair -P cmake/ThorWheelRelease.cmake
```

The release entry point invokes `auditwheel` and Twine through the release Python
interpreter (`python -m auditwheel`, `python -m twine`) rather than requiring
their console-script wrappers on `PATH`. In the Thor manylinux image it
automatically selects `/opt/python/cp312-cp312/bin/python`; override this with
`-DTHOR_RELEASE_PYTHON_EXECUTABLE=/path/to/python` when needed.

This is deliberately a source-tree CMake script rather than a target in the
scikit-build build directory. `python -m build` configures that directory using
a temporary isolated PEP 517 environment whose Ninja/CMake executables may no
longer exist after the build command returns.

This recreates `wheelhouse/`, runs `auditwheel repair` on each distribution,
keeps the NVIDIA CUDA/cuDNN user-space libraries external, explicitly keeps
`libThor.so` external to `thor-cuda`, and then verifies all of the release
invariants together: exact Thor/kernel versions, one backend per kernel wheel,
no backend in the central wheel, no bundled NVIDIA runtime libraries,
`manylinux_2_28_x86_64` tags, and the configured per-wheel PyPI size ceiling.
It finishes by running `twine check` on all three repaired wheels.

After installing and testing the repaired wheels, publication is also CMake
orchestrated. Credentials remain standard Twine configuration/environment:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD='...'
export THOR_PUBLISH_CONFIRM=YES
cmake -DTHOR_RELEASE_ACTION=publish -P cmake/ThorWheelRelease.cmake
```

The `publish` action recreates and re-verifies the wheelhouse before uploading in dependency
order: `thor-cuda-kernels-sm89`, then `thor-cuda-kernels-sm120`, then
`thor-cuda`. The public package is intentionally last because it requires both
kernel distributions at the exact same version. For TestPyPI, also export
`THOR_TWINE_REPOSITORY=testpypi` before invoking the publish target.
