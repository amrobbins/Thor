FROM quay.io/pypa/manylinux_2_28_x86_64:latest

ARG CUDA_MAJOR=13
ARG CUDA_MINOR=3

ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN dnf install -y \
      dnf-plugins-core \
      ca-certificates \
      curl \
      file \
      findutils \
      patchelf \
      which \
      binutils \
      git \
      ninja-build \
      cmake \
      libgcc \
      libstdc++ \
      libgomp \
      liburing \
      liburing-devel \
      openssl-devel \
    && dnf clean all

ARG NLOHMANN_JSON_VERSION=3.11.3

RUN curl -L \
      "https://github.com/nlohmann/json/releases/download/v${NLOHMANN_JSON_VERSION}/json.tar.xz" \
      -o /tmp/json.tar.xz && \
    mkdir -p /tmp/json-src && \
    tar -xf /tmp/json.tar.xz -C /tmp/json-src --strip-components=1 && \
    cmake -S /tmp/json-src -B /tmp/json-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DJSON_BuildTests=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/json-build --target install && \
    rm -rf /tmp/json-src /tmp/json-build /tmp/json.tar.xz

RUN dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf makecache

RUN dnf install -y \
      cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR} \
      cuda-libraries-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      cuda-cudart-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      cuda-nvrtc-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      libcublas-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      libnvjitlink-devel-${CUDA_MAJOR}-${CUDA_MINOR} \
      libcudnn9-devel-cuda-${CUDA_MAJOR} \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
ENV PATH="/opt/python/cp312-cp312/bin:${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN nvcc --version && \
    test -e "${CUDA_HOME}/lib64/libcudart.so" && \
    test -e "${CUDA_HOME}/lib64/libnvrtc.so" && \
    test -e "${CUDA_HOME}/lib64/libcublas.so" && \
    test -e "${CUDA_HOME}/lib64/libcublasLt.so" && \
    test -e "${CUDA_HOME}/lib64/libnvJitLink.so"

RUN /opt/python/cp312-cp312/bin/python -m pip install -U \
      pip \
      build \
      wheel \
      auditwheel \
      nanobind \
      pytest \
      numpy \
      ml_dtypes \
      twine \
      nvidia-cudnn-frontend==1.23.0

RUN printf '#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n' > /tmp/thor_openmp_smoke.cpp \
    && g++ -fopenmp /tmp/thor_openmp_smoke.cpp -o /tmp/thor_openmp_smoke \
    && /tmp/thor_openmp_smoke \
    && rm -f /tmp/thor_openmp_smoke.cpp /tmp/thor_openmp_smoke

RUN dnf install -y libarchive-devel

RUN dnf install -y ccache

# Mounted repo is owned by your host user, but container runs as root.
RUN git config --global --add safe.directory /io


#~/Thor$ sudo docker build -f docker/manylinux_2_28_cuda13.Dockerfile -t thor-manylinux_2_28-cuda13 .
#
#sudo docker run --rm -it --gpus all \
#  --privileged \
#  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
#  -v "$PWD":/io \
#  -w /io \
#  thor-manylinux_2_28-cuda13 \
#  bash

## Clean hard, so no local Ubuntu artifacts leak in.
#rm -rf build bindings/python/build bindings/python/dist bindings/python/wheelhouse
#find . -name '*.so' -path '*/build/*' -delete
#
## 1. native
#/opt/python/cp312-cp312/bin/python -m pip install -U pip setuptools wheel
#/opt/python/cp312-cp312/bin/python -m pip install \
#  nvidia-cuda-runtime==13.3.29 \
#  nvidia-cuda-nvrtc==13.3.33 \
#  nvidia-nvjitlink==13.4.52 \
#  nvidia-cublas==13.6.0.2 \
#  nvidia-cusparse==12.8.2.51 \
#  nvidia-cusolver==12.2.6.9 \
#  nvidia-cuda-cccl==13.3.3.4.1 \
#  nvidia-cudnn-cu13==9.26.0.51 \
#  nvidia-cudnn-frontend==1.29.0
#export CMAKE_GENERATOR=Ninja
#export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
#    -DTHOR_USE_PROJECT_VENV=OFF \
#    -DTHOR_PYTHON_EXECUTABLE=/opt/python/cp312-cp312/bin/python \
#    -DTHOR_PYTHON_AUTO_INSTALL=OFF"
#rm -rf cmake-build-release
#cmake -S . -B cmake-build-release \
#    -G Ninja \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DTHOR_USE_PROJECT_VENV=OFF \
#    -DTHOR_PYTHON_EXECUTABLE=/opt/python/cp312-cp312/bin/python \
#    -DTHOR_PYTHON_AUTO_INSTALL=OFF
#cmake --build cmake-build-release -j 32 && ctest --test-dir cmake-build-release --output-on-failure
#
## 2. wheel
#cd /io/bindings/python
#rm -rf build dist wheelhouse
#/opt/python/cp312-cp312/bin/python -m build --wheel
#export TWINE_USERNAME=__token__
#export TWINE_PASSWORD='pypi-redacted'
## One PEP 517 build produces the complete raw release set:
##   thor-cuda, thor-cuda-kernels-sm89, thor-cuda-kernels-sm120.
#/opt/python/cp312-cp312/bin/python -m twine check dist/*
#ls -lh dist/*.whl
#
## Repair and verify the three wheels as one release unit.  The CMake release
## gate excludes NVIDIA user-space libraries from every wheel and excludes
## libThor.so from thor-cuda because the selected kernel wheel supplies it at
## runtime.  It also enforces payload boundaries, exact kernel dependencies,
## manylinux_2_28 tags, and the per-file PyPI size ceiling.
#cd /io
#cmake -DTHOR_RELEASE_ACTION=repair -P cmake/ThorWheelRelease.cmake
#cd /io/bindings/python
#ls -lh wheelhouse/*.whl
#
## Test the wheel
#python - <<'PY'
#from pathlib import Path
#import zipfile
#
#wheel = next(Path("wheelhouse").glob("thor_cuda-*.whl"))
#with zipfile.ZipFile(wheel) as z:
#    metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
#    metadata = z.read(metadata_name).decode()
#    print(metadata_name)
#    for line in metadata.splitlines():
#        if line.startswith("Requires-Dist: nvidia") or line.startswith("Requires-Dist: cuda-toolkit"):
#            print(line)
#
#    resolved = z.read("thor/_cuda_stack_resolved.py").decode()
#    print("\n--- resolved stack ---")
#    for line in resolved.splitlines():
#        if "CudaDistribution(" in line:
#            print(line.strip())
#PY
#
#/opt/python/cp312-cp312/bin/python -m venv /tmp/thor-wheel-test
#source /tmp/thor-wheel-test/bin/activate
#python -m pip install -U pip pytest
#python -m pip install wheelhouse/*.whl
#cd /tmp
#python - <<'PY'
#import thor
#print("imported thor from:", thor.__file__)
#print("thor version:", getattr(thor, "__version__", "<missing>"))
#PY
#python -m pytest -q /io/bindings/python/test
#deactivate
#
#cd /io/bindings/python
#
## Publish only after the installed-wheel test passes.  The CMake target
## re-verifies the wheelhouse and uploads in dependency order:
##   1. thor-cuda-kernels-sm89
##   2. thor-cuda-kernels-sm120
##   3. thor-cuda
## Credentials remain ordinary Twine environment/configuration.  For TestPyPI
## additionally export THOR_TWINE_REPOSITORY=testpypi.
#export THOR_PUBLISH_CONFIRM=YES
#cd /io
#cmake -DTHOR_RELEASE_ACTION=publish -P cmake/ThorWheelRelease.cmake
