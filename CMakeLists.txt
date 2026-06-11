cmake_minimum_required(VERSION 3.22)

# ============================================================
# CUDA compiler bootstrap
# ============================================================
# CMake has to identify the CUDA compiler while project() is executing, so
# the compiler path and target architectures must be initialized before the
# CUDA language is enabled. CUDA 13.x no longer supports some older default
# probe architectures, and CMake's automatic architecture detection can fail
# before Thor's normal configuration code runs.
set(THOR_MIN_CUDA_TOOLKIT_VERSION 13.3)
set(
        THOR_CUDA_ARCHITECTURES
        "120"
        CACHE STRING
        "CUDA architectures for Thor AOT CUDA compilation. Default is sm_120 for RTX 50-series / Blackwell."
)
set_property(CACHE THOR_CUDA_ARCHITECTURES PROPERTY STRINGS 120)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${THOR_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
endif ()

# Do not require /usr/local/cuda to be a valid symlink. Prefer an explicitly
# requested compiler, then CUDA_HOME/CUDA_PATH, then the versioned CUDA 13.3
# install path, and finally PATH. Setting this before project() avoids CMake
# caching a stale /usr/local/cuda/bin/nvcc path.
if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    set(_thor_nvcc_hints "")
    if (DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
        list(APPEND _thor_nvcc_hints "$ENV{CUDA_HOME}/bin")
    endif ()
    if (DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
        list(APPEND _thor_nvcc_hints "$ENV{CUDA_PATH}/bin")
    endif ()
    list(APPEND _thor_nvcc_hints
            "/usr/local/cuda-13/bin"
            "/usr/local/cuda-13.3/bin"
            "/usr/local/cuda/bin"
    )

    find_program(_thor_detected_nvcc
            NAMES nvcc
            HINTS ${_thor_nvcc_hints}
            NO_DEFAULT_PATH
    )
    if (NOT _thor_detected_nvcc)
        find_program(_thor_detected_nvcc NAMES nvcc)
    endif ()
    if (_thor_detected_nvcc)
        set(CMAKE_CUDA_COMPILER "${_thor_detected_nvcc}" CACHE FILEPATH "CUDA compiler for Thor" FORCE)
    endif ()
endif ()

set(_thor_cuda_compiler_for_root "")
if (DEFINED CMAKE_CUDA_COMPILER AND CMAKE_CUDA_COMPILER)
    set(_thor_cuda_compiler_for_root "${CMAKE_CUDA_COMPILER}")
elseif (DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
    set(_thor_cuda_compiler_for_root "$ENV{CUDACXX}")
endif ()

if (_thor_cuda_compiler_for_root)
    get_filename_component(_thor_cuda_bin_dir "${_thor_cuda_compiler_for_root}" DIRECTORY)
    get_filename_component(_thor_cuda_root_dir "${_thor_cuda_bin_dir}" DIRECTORY)
    if (EXISTS "${_thor_cuda_root_dir}/include/cuda.h")
        if (NOT DEFINED CUDAToolkit_ROOT OR "${CUDAToolkit_ROOT}" STREQUAL "")
            set(CUDAToolkit_ROOT "${_thor_cuda_root_dir}" CACHE PATH "CUDA Toolkit root for Thor" FORCE)
        endif ()
        if (NOT DEFINED CUDA_TOOLKIT_ROOT_DIR OR "${CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "")
            set(CUDA_TOOLKIT_ROOT_DIR "${_thor_cuda_root_dir}" CACHE PATH "CUDA Toolkit root for legacy CMake find modules" FORCE)
        endif ()
    endif ()
endif ()

# Prefer Boost's own CMake config instead of the deprecated FindBoost module.
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif ()

# Read THOR_VERSION from ThorVersion.h.in
file(READ "${CMAKE_SOURCE_DIR}/ThorVersion.h.in" _thor_version_header)

# 1) Full version string, e.g. "0.1.1.dev0"
string(
        REGEX MATCH
        "#define[ \t]+THOR_VERSION[ \t]+\"([^\"]*)\""
        _thor_version_match
        "${_thor_version_header}"
)

if (NOT _thor_version_match)
    message(FATAL_ERROR "Could not find THOR_VERSION in ThorVersion.h.in")
endif ()

set(THOR_VERSION_FULL "${CMAKE_MATCH_1}")  # e.g. "0.1.1.dev0"

# 2) Numeric prefix for CMake project version, e.g. "0.1.1"
string(
        REGEX MATCH
        "^[0-9]+(\\.[0-9]+)*"
        _thor_version_numeric_match
        "${THOR_VERSION_FULL}"
)

if (NOT _thor_version_numeric_match)
    message(FATAL_ERROR "THOR_VERSION must start with numeric semver, got ${THOR_VERSION_FULL}")
endif ()

set(THOR_VERSION_NUMERIC "${CMAKE_MATCH_0}")  # "0.1.1"

# 3) Use numeric for CMake's project version
project(Thor VERSION ${THOR_VERSION_NUMERIC} LANGUAGES C CXX CUDA)


# ============================================================
# Thor Python selection
# ============================================================
option(THOR_USE_PROJECT_VENV "Use project .venv Python when present" ON)

set(
        THOR_PYTHON_EXECUTABLE
        ""
        CACHE FILEPATH
        "Python executable to use for Thor Python bindings. Overrides project .venv detection."
)

set(_thor_selected_python "")

if (THOR_PYTHON_EXECUTABLE)
    set(_thor_selected_python "${THOR_PYTHON_EXECUTABLE}")
elseif (DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
    set(_thor_selected_python "${Python_EXECUTABLE}")
elseif (DEFINED Python3_EXECUTABLE AND Python3_EXECUTABLE)
    set(_thor_selected_python "${Python3_EXECUTABLE}")
elseif (THOR_USE_PROJECT_VENV AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
    set(_thor_selected_python "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
endif ()

if (_thor_selected_python)
    set(Python_EXECUTABLE "${_thor_selected_python}" CACHE FILEPATH "Python executable for Thor bindings" FORCE)
    set(Python3_EXECUTABLE "${_thor_selected_python}" CACHE FILEPATH "Python3 executable for Thor/dependency CMake" FORCE)

    get_filename_component(_thor_python_prefix "${_thor_selected_python}" DIRECTORY)
    get_filename_component(_thor_python_prefix "${_thor_python_prefix}" DIRECTORY)

    set(Python_ROOT_DIR "${_thor_python_prefix}" CACHE PATH "Python root for Thor bindings" FORCE)
    set(Python3_ROOT_DIR "${_thor_python_prefix}" CACHE PATH "Python3 root for Thor/dependency CMake" FORCE)

    message(STATUS "Thor: selected Python = ${_thor_selected_python}")
endif ()

set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

# ============================================================
# CUDA 13.3 / CompileIQ settings
# ============================================================
set(THOR_CUDA_CXX_STANDARD 20 CACHE STRING "C++ dialect used for CUDA translation units and NVRTC generated kernels")
set_property(CACHE THOR_CUDA_CXX_STANDARD PROPERTY STRINGS 20)

set(
        THOR_CUDA_COMPILEIQ_ACF
        ""
        CACHE FILEPATH
        "Optional NVIDIA CompileIQ Advanced Controls File (.acf) passed to nvcc/ptxas via --apply-controls for Thor AOT CUDA sources."
)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS THOR_MIN_CUDA_TOOLKIT_VERSION)
    message(FATAL_ERROR
            "Thor now requires CUDA ${THOR_MIN_CUDA_TOOLKIT_VERSION}+ for CUDA 13.3 libraries, C++20 CUDA, and CompileIQ ACF support. "
            "Detected CUDA compiler ${CMAKE_CUDA_COMPILER_VERSION} at ${CMAKE_CUDA_COMPILER}.")
endif ()

message(STATUS "Thor: CUDA compiler = ${CMAKE_CUDA_COMPILER}")
message(STATUS "Thor: CUDA compiler version = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "Thor: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# Global settings
# ============================================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_CUDA_STANDARD ${THOR_CUDA_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")

# Default to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================
# Fast / multi-threaded linker selection
# ============================================================
set(THOR_LINKER "default" CACHE STRING "Linker to use: default, auto, mold, lld, gold, bfd")
set_property(CACHE THOR_LINKER PROPERTY STRINGS auto default mold lld gold bfd)

set(
        THOR_LINKER_THREADS
        ""
        CACHE STRING
        "Explicit linker thread count. Empty means use the selected linker's default."
)

set(_thor_selected_linker "")
set(_thor_selected_linker_dir "")
set(_thor_selected_linker_path "")

# GCC's -fuse-ld=<name> does not run a binary named <name>; it looks for
# linker-driver names such as ld.mold or ld.lld.  Some installs provide only
# mold/lld, so create a private build-tree symlink with the name GCC expects.
function(_thor_prepare_fuse_linker linker_name linker_driver_name result_var)
    set(_prepared_path "")
    set(_prepared_dir "")

    find_program(_driver_path NAMES "${linker_driver_name}" NO_CACHE)
    if (_driver_path)
        get_filename_component(_prepared_dir "${_driver_path}" DIRECTORY)
        set(_prepared_path "${_driver_path}")
    else ()
        find_program(_plain_path NAMES "${linker_name}" NO_CACHE)
        if (_plain_path)
            set(_prepared_dir "${CMAKE_BINARY_DIR}/thor-linker-bin")
            set(_prepared_path "${_prepared_dir}/${linker_driver_name}")
            file(MAKE_DIRECTORY "${_prepared_dir}")
            file(REMOVE "${_prepared_path}")
            execute_process(
                    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_plain_path}" "${_prepared_path}"
                    RESULT_VARIABLE _symlink_result
            )
            if (NOT _symlink_result EQUAL 0)
                set(_prepared_path "")
                set(_prepared_dir "")
            endif ()
        endif ()
    endif ()

    set(${result_var} "${_prepared_path}" PARENT_SCOPE)
    set(${result_var}_DIR "${_prepared_dir}" PARENT_SCOPE)
endfunction()

if (THOR_LINKER STREQUAL "default" OR THOR_LINKER STREQUAL "")
    # Preserve the compiler driver's original behavior. Do not add -fuse-ld,
    # do not add -B, and do not try to force ld.bfd/gold/lld/mold.
    message(STATUS "Thor: using compiler default linker")
elseif (THOR_LINKER STREQUAL "auto")
    # Fast-linker auto mode is opt-in. Prefer mold, then lld, then gold;
    # if none are available, leave the compiler default linker unchanged.
    _thor_prepare_fuse_linker("mold" "ld.mold" _thor_mold_path)
    if (_thor_mold_path)
        set(_thor_selected_linker "mold")
        set(_thor_selected_linker_path "${_thor_mold_path}")
        set(_thor_selected_linker_dir "${_thor_mold_path_DIR}")
    else ()
        _thor_prepare_fuse_linker("lld" "ld.lld" _thor_lld_path)
        if (_thor_lld_path)
            set(_thor_selected_linker "lld")
            set(_thor_selected_linker_path "${_thor_lld_path}")
            set(_thor_selected_linker_dir "${_thor_lld_path_DIR}")
        else ()
            find_program(_thor_ld_gold NAMES ld.gold NO_CACHE)
            if (_thor_ld_gold)
                get_filename_component(_thor_selected_linker_dir "${_thor_ld_gold}" DIRECTORY)
                set(_thor_selected_linker "gold")
                set(_thor_selected_linker_path "${_thor_ld_gold}")
            else ()
                message(STATUS "Thor: no mold/lld/gold found; using compiler default linker")
            endif ()
        endif ()
    endif ()
elseif (THOR_LINKER STREQUAL "mold")
    _thor_prepare_fuse_linker("mold" "ld.mold" _thor_mold_path)
    if (NOT _thor_mold_path)
        message(FATAL_ERROR
                "THOR_LINKER=mold was requested, but neither ld.mold nor mold was found.\n"
                "Install mold, or configure with -DTHOR_LINKER=lld/default.")
    endif ()
    set(_thor_selected_linker "mold")
    set(_thor_selected_linker_path "${_thor_mold_path}")
    set(_thor_selected_linker_dir "${_thor_mold_path_DIR}")
elseif (THOR_LINKER STREQUAL "lld")
    _thor_prepare_fuse_linker("lld" "ld.lld" _thor_lld_path)
    if (NOT _thor_lld_path)
        message(FATAL_ERROR
                "THOR_LINKER=lld was requested, but neither ld.lld nor lld was found.\n"
                "Install lld, or configure with -DTHOR_LINKER=mold/default.")
    endif ()
    set(_thor_selected_linker "lld")
    set(_thor_selected_linker_path "${_thor_lld_path}")
    set(_thor_selected_linker_dir "${_thor_lld_path_DIR}")
elseif (THOR_LINKER STREQUAL "gold")
    find_program(_thor_ld_gold NAMES ld.gold NO_CACHE)
    if (NOT _thor_ld_gold)
        message(FATAL_ERROR "THOR_LINKER=gold was requested, but ld.gold was not found.")
    endif ()
    get_filename_component(_thor_selected_linker_dir "${_thor_ld_gold}" DIRECTORY)
    set(_thor_selected_linker "gold")
    set(_thor_selected_linker_path "${_thor_ld_gold}")
elseif (THOR_LINKER STREQUAL "bfd")
    find_program(_thor_ld_bfd NAMES ld.bfd NO_CACHE)
    if (_thor_ld_bfd)
        get_filename_component(_thor_selected_linker_dir "${_thor_ld_bfd}" DIRECTORY)
        set(_thor_selected_linker_path "${_thor_ld_bfd}")
    endif ()
    set(_thor_selected_linker "bfd")
else ()
    message(FATAL_ERROR "Invalid THOR_LINKER='${THOR_LINKER}'. Use auto, default, mold, lld, gold, or bfd.")
endif ()

if (_thor_selected_linker)
    if (_thor_selected_linker_path)
        message(STATUS "Thor: selected linker = ${_thor_selected_linker} (${_thor_selected_linker_path})")
    else ()
        message(STATUS "Thor: selected linker = ${_thor_selected_linker}")
    endif ()
endif ()

function(thor_apply_linker_settings target_name)
    if (NOT TARGET ${target_name})
        return()
    endif ()

    if (_thor_selected_linker)
        if (_thor_selected_linker_dir)
            # Help GCC find ld.mold/ld.lld/ld.gold even when it is not in GCC's
            # built-in linker search directory.
            target_link_options(${target_name} PRIVATE
                    "-B${_thor_selected_linker_dir}/"
            )
        endif ()

        target_link_options(${target_name} PRIVATE
                "-fuse-ld=${_thor_selected_linker}"
        )
    endif ()

    if (_thor_selected_linker STREQUAL "mold")
        if (THOR_LINKER_THREADS)
            target_link_options(${target_name} PRIVATE
                    "LINKER:--thread-count=${THOR_LINKER_THREADS}"
            )
        endif ()
    elseif (_thor_selected_linker STREQUAL "lld")
        if (THOR_LINKER_THREADS)
            target_link_options(${target_name} PRIVATE
                    "LINKER:--threads=${THOR_LINKER_THREADS}"
            )
        endif ()
    elseif (_thor_selected_linker STREQUAL "gold")
        target_link_options(${target_name} PRIVATE
                "LINKER:--threads"
        )
        if (THOR_LINKER_THREADS)
            target_link_options(${target_name} PRIVATE
                    "LINKER:--thread-count=${THOR_LINKER_THREADS}"
            )
        endif ()
    elseif (_thor_selected_linker STREQUAL "bfd" AND THOR_LINKER_THREADS)
        message(WARNING
                "Thor: ld.bfd does not provide a general multi-threaded link option; "
                "THOR_LINKER_THREADS will be ignored."
        )
    endif ()
endfunction()

# ============================================================
# Release-only link-time optimization
# ============================================================
# Host C++ LTO is handled through CMake's IPO support so CMake can pick the
# correct compiler/linker flags for the active C++ toolchain. Keep this
# target-scoped rather than setting CMAKE_INTERPROCEDURAL_OPTIMIZATION globally
# so third-party FetchContent targets are not accidentally changed.
set(_thor_release_lto_configured FALSE)
if (CMAKE_CONFIGURATION_TYPES)
    set(_thor_release_lto_configured TRUE)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_thor_release_lto_configured TRUE)
endif ()

if (_thor_release_lto_configured)
    include(CheckIPOSupported)
    check_ipo_supported(
            RESULT _thor_cxx_ipo_supported
            OUTPUT _thor_cxx_ipo_check_output
            LANGUAGES CXX
    )
    if (NOT _thor_cxx_ipo_supported)
        message(FATAL_ERROR
                "Thor Release builds require C++ IPO/LTO, but CMake reported that it is not supported by the current toolchain/linker:\n"
                "${_thor_cxx_ipo_check_output}")
    endif ()
endif ()

function(thor_enable_release_host_lto target_name)
    if (NOT TARGET ${target_name})
        return()
    endif ()

    # Release-only host-side C++ LTO. Debug and RelWithDebInfo remain unchanged.
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endfunction()

function(thor_enable_release_no_semantic_interposition target_name)
    if (NOT TARGET ${target_name})
        return()
    endif ()

    # Release-only host-code optimization for ELF shared-library builds.  This
    # lets the compiler assume definitions within the current linkage unit are
    # not semantically replaced by another DSO definition at runtime.  Apply it
    # to C++ translation units directly and to CUDA translation units through
    # nvcc's host-compiler forwarding path.
    target_compile_options(${target_name} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-fno-semantic-interposition>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fno-semantic-interposition>
    )

    # With IPO/LTO, keep the option visible to the host link/LTO step too.
    # HOST_LINK prevents the flag from being added to CUDA device-link commands.
    target_link_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:$<HOST_LINK:-fno-semantic-interposition>>
    )
endfunction()

function(thor_enable_release_cuda_device_lto target_name)
    if (NOT TARGET ${target_name})
        return()
    endif ()

    # nvcc device LTO requires relocatable device code plus a device-link step.
    # CUDA_SEPARABLE_COMPILATION is target-wide in CMake, so for normal
    # single-config builds this is enabled only when the configured build type is
    # Release. Multi-config generators need the device-link path available for
    # the Release configuration, but the actual -dlto flags below are still
    # Release-only.
    if (CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE STREQUAL "Release")
        set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif ()

    # CMake's INTERPROCEDURAL_OPTIMIZATION_RELEASE property emits the
    # compile-side CUDA LTO IR by adding lto_<arch> code generation targets
    # such as code=[compute_120,lto_120].  Do not also pass compile-time -dlto
    # here: nvcc rejects -dlto together with explicit -gencode/code control and
    # tells callers to use code=lto_<arch> for the compile-side intermediate.
    # Keep only the Release device-link -dlto flag so nvlink consumes and
    # optimizes the LTO IR emitted by the compile step.
    target_link_options(${target_name} PRIVATE
            "$<$<CONFIG:Release>:$<DEVICE_LINK:-dlto>>"
    )
endfunction()

# ============================================================
# Automatic Git versioning
# ============================================================
set(
        THOR_GIT_VERSION_MODE
        "auto"
        CACHE STRING
        "Git version mode: auto embeds git describe output; unknown always embeds 'unknown' and does not track .git files."
)
set_property(CACHE THOR_GIT_VERSION_MODE PROPERTY STRINGS auto unknown)

if (THOR_GIT_VERSION_MODE STREQUAL "unknown")
    set(GIT_VERSION "unknown")
elseif (THOR_GIT_VERSION_MODE STREQUAL "auto")
    set(_thor_git_describe_result 1)
    find_package(Git QUIET)
    if (Git_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tags
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_VERSION
                RESULT_VARIABLE _thor_git_describe_result
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif ()

    if (NOT Git_FOUND OR NOT _thor_git_describe_result EQUAL 0 OR GIT_VERSION STREQUAL "")
        set(GIT_VERSION "unknown")
    endif ()

    if (EXISTS "${CMAKE_SOURCE_DIR}/.git/HEAD")
        # Reconfigure only when embedding a real git version. Development builds
        # can use -DTHOR_GIT_VERSION_MODE=unknown to keep generated/ThorVersion.h
        # stable across commits, pushes, dirty-state changes, and index updates.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                "${CMAKE_SOURCE_DIR}/.git/HEAD"
                "${CMAKE_SOURCE_DIR}/.git/index"
                "${CMAKE_SOURCE_DIR}/.git/packed-refs"
        )

        file(READ "${CMAKE_SOURCE_DIR}/.git/HEAD" _git_head LIMIT 200)
        if (_git_head MATCHES "^ref: (.+)$")
            set(_git_ref "${CMAKE_SOURCE_DIR}/.git/${CMAKE_MATCH_1}")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_git_ref}")
        endif ()
    endif ()
else ()
    message(FATAL_ERROR "Invalid THOR_GIT_VERSION_MODE='${THOR_GIT_VERSION_MODE}'. Use auto or unknown.")
endif ()

message(STATUS "Git version (${THOR_GIT_VERSION_MODE}): ${GIT_VERSION}")

configure_file(
        ${CMAKE_SOURCE_DIR}/ThorVersion.h.in
        ${CMAKE_BINARY_DIR}/generated/ThorVersion.h
        @ONLY
)

# ============================================================
# Source files
# ============================================================
file(GLOB_RECURSE THOR_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/Utilities/**/*.cpp"
        "${CMAKE_SOURCE_DIR}/Utilities/**/*.cu"
        "${CMAKE_SOURCE_DIR}/DeepLearning/**/*.cpp"
        "${CMAKE_SOURCE_DIR}/DeepLearning/**/*.cu"
        #        "${CMAKE_SOURCE_DIR}/Demos/**/*.cpp"
        #        "${CMAKE_SOURCE_DIR}/Demos/**/*.cu"
)

# ============================================================
# Create shared library
# ============================================================
#add_library(Thor STATIC ${THOR_SOURCES})
add_library(Thor SHARED ${THOR_SOURCES})
set_target_properties(Thor PROPERTIES
        CXX_STANDARD 20
        CUDA_STANDARD ${THOR_CUDA_CXX_STANDARD}
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

target_include_directories(Thor
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)

# ============================================================
# Compiler and linker flags (replacing Makefile logic)
# ============================================================
# Common flags
target_compile_options(Thor PRIVATE
        -fPIC
        -Wall
        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        -Wdeprecated
)
target_link_options(Thor PRIVATE
        "LINKER:--no-as-needed"
)
thor_apply_linker_settings(Thor)
thor_enable_release_host_lto(Thor)
thor_enable_release_no_semantic_interposition(Thor)
thor_enable_release_cuda_device_lto(Thor)

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(Thor PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit ${THOR_MIN_CUDA_TOOLKIT_VERSION} REQUIRED)
find_package(OpenSSL REQUIRED)

# Main CUDA headers directory for things like cuda.h.
# CUDAToolkit_INCLUDE_DIRS is a list; take the primary include dir and
# normalize it so CMake never treats it as relative.
list(GET CUDAToolkit_INCLUDE_DIRS 0 THOR_CUDA_INCLUDE_DIR)
cmake_path(ABSOLUTE_PATH THOR_CUDA_INCLUDE_DIR NORMALIZE)

if (NOT EXISTS "${THOR_CUDA_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR
            "Resolved CUDA include dir is invalid: ${THOR_CUDA_INCLUDE_DIR}")
endif ()

target_link_libraries(Thor PRIVATE
        OpenSSL::Crypto
        CUDA::cudart
        # CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cusolver
        CUDA::cusparse
        CUDA::nvrtc
)

# Find cuDNN without hard-coding platform-specific library directories.
# Prefer the detected CUDA toolkit library locations, then fall back to the
# normal system library search path.
find_library(THOR_CUDNN_LIBRARY
        NAMES cudnn
        HINTS
        "${CUDAToolkit_LIBRARY_DIR}"
        "${CUDAToolkit_TARGET_DIR}/lib"
        "${CUDAToolkit_TARGET_DIR}/lib64"
)
if (NOT THOR_CUDNN_LIBRARY)
    find_library(THOR_CUDNN_LIBRARY NAMES cudnn REQUIRED)
endif ()

target_link_libraries(Thor PRIVATE
        "${THOR_CUDNN_LIBRARY}"
)

# CCCL provides the CUB headers. CUDA 13 wheels normally use the unified
# nvidia/cu13/include tree, but older/split wheel layouts may use
# nvidia/cuda_cccl/include. Default to the main CUDA include dir, then prefer
# an explicitly configured or discoverable CCCL include root when available.
set(THOR_CUDA_CCCL_INCLUDE_DIR "${THOR_CUDA_INCLUDE_DIR}" CACHE PATH "CUDA CCCL/CUB include directory")

if (DEFINED ENV{THOR_CUDA_CCCL_INCLUDE_DIR} AND NOT "$ENV{THOR_CUDA_CCCL_INCLUDE_DIR}" STREQUAL "")
    set(THOR_CUDA_CCCL_INCLUDE_DIR "$ENV{THOR_CUDA_CCCL_INCLUDE_DIR}" CACHE PATH "CUDA CCCL/CUB include directory" FORCE)
elseif (DEFINED Python3_EXECUTABLE AND Python3_EXECUTABLE)
    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
            "import sysconfig
from pathlib import Path
site = Path(sysconfig.get_paths()['platlib'])
candidates = [site / 'nvidia' / 'cu13' / 'include', site / 'nvidia' / 'cuda_cccl' / 'include']
print(next((str(c) for c in candidates if (c / 'cub' / 'cub.cuh').exists()), ''))"
            OUTPUT_VARIABLE _thor_python_cccl_include_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (_thor_python_cccl_include_dir)
        set(THOR_CUDA_CCCL_INCLUDE_DIR "${_thor_python_cccl_include_dir}" CACHE PATH "CUDA CCCL/CUB include directory" FORCE)
    endif ()
endif ()

target_include_directories(Thor PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        "${THOR_CUDA_CCCL_INCLUDE_DIR}"
)

target_compile_definitions(Thor PRIVATE
        THOR_CUDA_INCLUDE_DIR=\"${THOR_CUDA_INCLUDE_DIR}\"
        THOR_CUDA_CCCL_INCLUDE_DIR=\"${THOR_CUDA_CCCL_INCLUDE_DIR}\"
)

find_package(Curses REQUIRED)
target_include_directories(Thor PUBLIC ${CURSES_INCLUDE_DIRS})
target_link_libraries(Thor PUBLIC ${CURSES_LIBRARIES})

# ============================================================
# Boost
# ============================================================
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_NO_BOOST_CMAKE ON)

set(Boost_MIN_VERSION 1.66)

# Use Boost's config package (BoostConfig.cmake), not the old FindBoost module.
find_package(Boost ${Boost_MIN_VERSION} REQUIRED COMPONENTS
        serialization
        system
)
# Header-only: Boost::boost / Boost::headers gives you the include dirs.
target_link_libraries(Thor PRIVATE Boost::boost)

# Per-configuration flags
target_compile_options(Thor PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3 -ggdb -fsanitize=undefined>
        $<$<CONFIG:Debug>:-O0 -ggdb>
)

target_compile_definitions(Thor PUBLIC
        $<$<CONFIG:Release>:THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64>
        $<$<CONFIG:RelWithDebInfo>:THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64>
        $<$<CONFIG:Debug>:THOR_DEBUG GDK_NVDIRECT _FILE_OFFSET_BITS=64> # _GLIBCXX_DEBUG
)

# CUDA-specific extra flags
set(_thor_cuda_compile_options
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings>
)

if (THOR_CUDA_COMPILEIQ_ACF)
    cmake_path(ABSOLUTE_PATH THOR_CUDA_COMPILEIQ_ACF NORMALIZE)
    if (NOT EXISTS "${THOR_CUDA_COMPILEIQ_ACF}")
        message(FATAL_ERROR "THOR_CUDA_COMPILEIQ_ACF does not exist: ${THOR_CUDA_COMPILEIQ_ACF}")
    endif ()
    message(STATUS "Thor: applying CompileIQ ACF to AOT CUDA compilation: ${THOR_CUDA_COMPILEIQ_ACF}")
    list(APPEND _thor_cuda_compile_options
            $<$<COMPILE_LANGUAGE:CUDA>:--apply-controls=${THOR_CUDA_COMPILEIQ_ACF}>
    )
else ()
    message(STATUS "Thor: CompileIQ ACF not configured; pass -DTHOR_CUDA_COMPILEIQ_ACF=/path/to/file.acf after tuning to enable it for AOT CUDA sources")
endif ()

target_compile_options(Thor PRIVATE ${_thor_cuda_compile_options})

option(THOR_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" OFF)
if (THOR_ENABLE_SANITIZERS)
    add_compile_options(
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address,undefined>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fno-omit-frame-pointer>
    )
    add_link_options(
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    )
endif ()

# Tar file support
find_package(LibArchive REQUIRED)
target_link_libraries(Thor PRIVATE LibArchive::LibArchive)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
target_link_libraries(Thor PRIVATE PkgConfig::LIBURING)


# ============================================================
# Dependencies: nlohmann/json, pthread, dl
# ============================================================
find_package(nlohmann_json 3 CONFIG REQUIRED)
target_link_libraries(Thor PUBLIC nlohmann_json::nlohmann_json)

# Link dependencies
target_link_libraries(Thor PRIVATE
        pthread
        dl
)


# ============================================================
# GoogleTest
# ============================================================
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

# For fast CRC (REQUIRE zlib-ng native API)
include(FetchContent)
include(CheckCXXSourceCompiles)

FetchContent_Declare(zlibng
        GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng.git
        GIT_TAG stable
)

# zlib-ng options (set BEFORE MakeAvailable)
set(ZLIB_COMPAT OFF CACHE BOOL "" FORCE)        # want <zlib-ng.h> + zng_* API
set(ZLIB_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

# Important for manylinux wheels: avoid libz-ng.so.2 / ZLIB_NG_2.0.0 dynamic dependency.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Compile the CLMUL/VPCLMUL codepaths (runtime dispatch still happens)
set(WITH_PCLMULQDQ ON CACHE BOOL "" FORCE)
set(WITH_VPCLMULQDQ ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(zlibng)

# Find the actual target name exported by zlib-ng
set(ZLIBNG_TARGET "")
foreach (tgt IN ITEMS
        zlibstatic-ng
        zlibstatic
        zlib-ng::zlib-ng-static
        zlib-ng::zlib-ng
        zlib-ng
)
    if (TARGET ${tgt})
        set(ZLIBNG_TARGET ${tgt})
        break()
    endif ()
endforeach ()

get_target_property(_zng_type ${ZLIBNG_TARGET} TYPE)
message(STATUS "Thor: using zlib-ng target ${ZLIBNG_TARGET} of type ${_zng_type}")

if (_zng_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
            "Thor wheel builds must not link shared zlib-ng because auditwheel rejects ZLIB_NG_* "
            "symbol versions. Use a static zlib-ng target instead.")
endif ()

# ---- Hard guarantee at CONFIGURE time (compile-only checks, no linking to CMake targets) ----
# zlib-ng.h is often generated into the zlib-ng *build* dir, so include both.
set(THOR_ZLIBNG_INCLUDE_DIRS
        "${zlibng_BINARY_DIR}"
        "${zlibng_SOURCE_DIR}"
)

set(CMAKE_REQUIRED_INCLUDES "${THOR_ZLIBNG_INCLUDE_DIRS}")

# 1) Ensure this really is zlib-ng (recommended compile-time distinguisher)
check_cxx_source_compiles([=[
  #include <zlib-ng.h>
  #ifndef ZLIBNG_VERSION
  #error "Not zlib-ng"
  #endif
  int main() { return 0; }
]=] THOR_ZLIBNG_HEADER_OK)

if (NOT THOR_ZLIBNG_HEADER_OK)
    message(FATAL_ERROR
            "zlib-ng requirement failed: <zlib-ng.h> not from zlib-ng (ZLIBNG_VERSION missing). "
            "Check include paths; we expect zlib-ng from FetchContent.")
endif ()

# 2) Prefer zng_crc32_z (size_t len). If not present, accept zng_crc32.
check_cxx_source_compiles([=[
  #include <zlib-ng.h>
  #include <cstddef>
  #include <cstdint>
  #ifndef ZLIBNG_VERSION
  #error "Not zlib-ng"
  #endif
  using r = decltype(zng_crc32_z((uint32_t)0, (const uint8_t*)0, (size_t)0));
  int main() { return 0; }
]=] THOR_ZLIBNG_HAS_CRC32_Z)

if (NOT THOR_ZLIBNG_HAS_CRC32_Z)
    check_cxx_source_compiles([=[
    #include <zlib-ng.h>
    #include <cstdint>
    #ifndef ZLIBNG_VERSION
    #error "Not zlib-ng"
    #endif
    using r = decltype(zng_crc32((uint32_t)0, (const uint8_t*)0, (unsigned)0));
    int main() { return 0; }
  ]=] THOR_ZLIBNG_HAS_CRC32)
endif ()

if (NOT THOR_ZLIBNG_HAS_CRC32_Z AND NOT THOR_ZLIBNG_HAS_CRC32)
    message(FATAL_ERROR
            "zlib-ng requirement failed: header is zlib-ng, but neither zng_crc32_z nor zng_crc32 is declared.")
endif ()

# Convert to 0/1 for C/C++ preprocessor
set(THOR_ZLIBNG_HAS_CRC32_Z_INT 0)
if (THOR_ZLIBNG_HAS_CRC32_Z)
    set(THOR_ZLIBNG_HAS_CRC32_Z_INT 1)
endif ()

target_link_libraries(Thor PRIVATE ${ZLIBNG_TARGET})

target_include_directories(Thor PRIVATE ${THOR_ZLIBNG_INCLUDE_DIRS})

# Hard compile-time contract + feature flag
target_compile_definitions(Thor PRIVATE
        THOR_REQUIRE_ZLIBNG=1
        THOR_ZLIBNG_HAS_CRC32_Z=${THOR_ZLIBNG_HAS_CRC32_Z_INT}
)


file(GLOB_RECURSE TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp")
add_executable(thor_tests ${TEST_SOURCES})
thor_apply_linker_settings(thor_tests)
thor_enable_release_host_lto(thor_tests)
thor_enable_release_no_semantic_interposition(thor_tests)
target_link_libraries(thor_tests PRIVATE Thor GTest::gtest_main CUDA::cudart CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cusparse)
target_link_libraries(thor_tests PRIVATE "${THOR_CUDNN_LIBRARY}")
find_package(OpenMP REQUIRED)
target_link_libraries(thor_tests PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(thor_tests PRIVATE Boost::boost)
target_link_libraries(thor_tests PRIVATE PkgConfig::LIBURING)   # FIXME: I think I can remove this once UringDirect.cpp
include(GoogleTest)
gtest_discover_tests(thor_tests)

# Per-configuration flags
target_compile_definitions(thor_tests PRIVATE
        THOR_GTEST
        SOURCE_DIR="$<SHELL_PATH:${CMAKE_SOURCE_DIR}>"
)

target_compile_options(thor_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:Debug>:-O0 -ggdb>
)

add_executable(thor_embedding_sparse_backward_profile EXCLUDE_FROM_ALL
        benchmarks/EmbeddingSparseBackwardProfile.cpp
)
thor_apply_linker_settings(thor_embedding_sparse_backward_profile)
thor_enable_release_host_lto(thor_embedding_sparse_backward_profile)
thor_enable_release_no_semantic_interposition(thor_embedding_sparse_backward_profile)
target_link_libraries(thor_embedding_sparse_backward_profile PRIVATE Thor CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cusparse)
target_link_libraries(thor_embedding_sparse_backward_profile PRIVATE "${THOR_CUDNN_LIBRARY}")
target_link_libraries(thor_embedding_sparse_backward_profile PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(thor_embedding_sparse_backward_profile PRIVATE Boost::boost)
target_link_libraries(thor_embedding_sparse_backward_profile PRIVATE PkgConfig::LIBURING)
target_compile_options(thor_embedding_sparse_backward_profile PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:Debug>:-O0 -ggdb>
)

# ============================================================
# Install & Export for find_package()
# ============================================================
if (NOT DEFINED SKBUILD_PROJECT_NAME)
    install(TARGETS Thor
            EXPORT ThorTargets
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
    )
    install(FILES
            "${CMAKE_BINARY_DIR}/generated/Thor.h"
            "${CMAKE_BINARY_DIR}/generated/ThorVersion.h"
            DESTINATION include
    )

    install(EXPORT ThorTargets
            FILE ThorTargets.cmake
            NAMESPACE Thor::
            DESTINATION lib/cmake/Thor
    )

    install(FILES
            ${CMAKE_BINARY_DIR}/ThorConfig.cmake
            ${CMAKE_BINARY_DIR}/ThorConfigVersion.cmake
            DESTINATION lib/cmake/Thor
    )
endif ()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${CMAKE_BINARY_DIR}/ThorConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
        ${CMAKE_SOURCE_DIR}/cmake/ThorConfig.cmake.in
        ${CMAKE_BINARY_DIR}/ThorConfig.cmake
        INSTALL_DESTINATION lib/cmake/Thor
)

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-msse4.2" HAS_MSSE42)
if (HAS_MSSE42)
    set_source_files_properties(
            Utilities/TarFile/Crc32c.cpp
            PROPERTIES COMPILE_OPTIONS "-msse4.2"
    )
endif ()

# ============================================================
# Build the header generation tool
# ============================================================
add_executable(createMasterHeader buildTools/createMasterHeader.cpp)
set_target_properties(createMasterHeader PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/buildTools
)

# ============================================================
# Generate headerlist.txt
# ============================================================
# The umbrella header depends only on the set of public headers, not on every
# C++/CUDA source file.  Keeping headerlist.txt as a configure-time file avoids
# rewriting generated/Thor.h after ordinary .cpp/.cu edits, which otherwise
# causes broad rebuilds of files that include Thor.h.
file(GLOB_RECURSE THOR_PUBLIC_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/DeepLearning/*.h"
        "${CMAKE_SOURCE_DIR}/Utilities/*.h"
)

set(_thor_headerlist_entries "")
foreach (_thor_header IN LISTS THOR_PUBLIC_HEADERS)
    file(RELATIVE_PATH _thor_rel_header "${CMAKE_SOURCE_DIR}" "${_thor_header}")
    list(APPEND _thor_headerlist_entries "./${_thor_rel_header}")
endforeach ()
list(SORT _thor_headerlist_entries)

set(_thor_headerlist_contents "")
foreach (_thor_header IN LISTS _thor_headerlist_entries)
    string(APPEND _thor_headerlist_contents "${_thor_header}\n")
endforeach ()

set(_thor_headerlist_path "${CMAKE_BINARY_DIR}/headerlist.txt")
set(_thor_write_headerlist TRUE)
if (EXISTS "${_thor_headerlist_path}")
    file(READ "${_thor_headerlist_path}" _thor_existing_headerlist)
    if (_thor_existing_headerlist STREQUAL _thor_headerlist_contents)
        set(_thor_write_headerlist FALSE)
    endif ()
endif ()

if (_thor_write_headerlist)
    file(WRITE "${_thor_headerlist_path}" "${_thor_headerlist_contents}")
endif ()

# ============================================================
# Generate Thor.h from headerlist.txt
# ============================================================
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/generated/Thor.h
        COMMAND ${CMAKE_BINARY_DIR}/buildTools/createMasterHeader ${CMAKE_BINARY_DIR}/headerlist.txt
        DEPENDS createMasterHeader ${CMAKE_BINARY_DIR}/headerlist.txt
        COMMENT "Generating Thor.h from header list"
)

add_custom_target(ThorHeader ALL DEPENDS ${CMAKE_BINARY_DIR}/generated/Thor.h)
add_dependencies(Thor ThorHeader)


find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        RESULT_VARIABLE _thor_nanobind_result
        ERROR_VARIABLE _thor_nanobind_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT _thor_nanobind_result EQUAL 0)
    message(FATAL_ERROR
            "Failed to locate nanobind using Python executable:\n"
            "  ${Python_EXECUTABLE}\n"
            "stderr:\n"
            "${_thor_nanobind_error}\n"
            "Install nanobind into that Python environment."
    )
endif ()

list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

option(THOR_BUILD_PYTHON "Build Python bindings" ON)
if (THOR_BUILD_PYTHON)
    add_subdirectory(bindings/python)

    if (TARGET thor_py_deps)
        add_dependencies(Thor thor_py_deps)
    endif ()
endif ()

# For convenience, rebuild the python bindings before rebuilding the c++ tests
add_dependencies(thor_tests _thor)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.metadata as md
from pathlib import Path
try:
    dist = md.distribution('nvidia-cudnn-frontend')
    hits = [Path(dist.locate_file(f)).parent for f in (dist.files or []) if f.name == 'cudnn_frontend.h']
    print(hits[0] if hits else '')
except Exception:
    print('')"
        OUTPUT_VARIABLE THOR_CUDNN_FRONTEND_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT THOR_CUDNN_FRONTEND_INCLUDE_DIR)
    message(FATAL_ERROR
            "Thor attention requires nvidia-cudnn-frontend and cudnn_frontend.h. "
            "Install it with: ${Python3_EXECUTABLE} -m pip install nvidia-cudnn-frontend==1.23.0")
endif ()

message(STATUS "Thor: cuDNN Frontend include = ${THOR_CUDNN_FRONTEND_INCLUDE_DIR}")

target_include_directories(Thor PRIVATE "${THOR_CUDNN_FRONTEND_INCLUDE_DIR}")
