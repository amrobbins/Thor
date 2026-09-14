cmake_minimum_required(VERSION 3.28)

if (NOT DEFINED THOR_WHEEL_DIST_DIR OR THOR_WHEEL_DIST_DIR STREQUAL "")
    message(FATAL_ERROR "RepairThorWheels.cmake requires -DTHOR_WHEEL_DIST_DIR=/path/to/dist")
endif ()
if (NOT DEFINED THOR_WHEELHOUSE_DIR OR THOR_WHEELHOUSE_DIR STREQUAL "")
    message(FATAL_ERROR "RepairThorWheels.cmake requires -DTHOR_WHEELHOUSE_DIR=/path/to/wheelhouse")
endif ()
if (NOT DEFINED THOR_MANYLINUX_PLATFORM OR THOR_MANYLINUX_PLATFORM STREQUAL "")
    set(THOR_MANYLINUX_PLATFORM "manylinux_2_28_x86_64")
endif ()
if (NOT DEFINED THOR_PYPI_MAX_WHEEL_BYTES OR THOR_PYPI_MAX_WHEEL_BYTES STREQUAL "")
    set(THOR_PYPI_MAX_WHEEL_BYTES 104857600)
endif ()

get_filename_component(THOR_WHEEL_DIST_DIR "${THOR_WHEEL_DIST_DIR}" ABSOLUTE)
get_filename_component(THOR_WHEELHOUSE_DIR "${THOR_WHEELHOUSE_DIR}" ABSOLUTE)
get_filename_component(_thor_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

file(READ "${_thor_repo_root}/ThorVersion.h.in" _thor_version_text)
string(REGEX MATCH "#define[ \t]+THOR_VERSION[ \t]+\"v?([^\"]+)\"" _thor_version_match "${_thor_version_text}")
if (NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Unable to read THOR_VERSION from ${_thor_repo_root}/ThorVersion.h.in")
endif ()
set(_thor_version "${CMAKE_MATCH_1}")

function(_thor_require_single_raw_wheel pattern label out_var)
    file(GLOB _matches LIST_DIRECTORIES FALSE "${pattern}")
    list(LENGTH _matches _count)
    if (NOT _count EQUAL 1)
        message(FATAL_ERROR
                "Expected exactly one raw ${label} wheel matching:\n"
                "  ${pattern}\n"
                "Found ${_count}: ${_matches}")
    endif ()
    list(GET _matches 0 _wheel)
    set(${out_var} "${_wheel}" PARENT_SCOPE)
endfunction()

if (NOT IS_DIRECTORY "${THOR_WHEEL_DIST_DIR}")
    message(FATAL_ERROR "Thor raw wheel directory does not exist: ${THOR_WHEEL_DIST_DIR}")
endif ()

_thor_require_single_raw_wheel(
        "${THOR_WHEEL_DIST_DIR}/thor_cuda-${_thor_version}-*.whl"
        "thor-cuda"
        _thor_core_wheel
)
_thor_require_single_raw_wheel(
        "${THOR_WHEEL_DIST_DIR}/thor_cuda_kernels_sm89-${_thor_version}-*.whl"
        "thor-cuda-kernels-sm89"
        _thor_sm89_wheel
)
_thor_require_single_raw_wheel(
        "${THOR_WHEEL_DIST_DIR}/thor_cuda_kernels_sm120-${_thor_version}-*.whl"
        "thor-cuda-kernels-sm120"
        _thor_sm120_wheel
)

if (NOT DEFINED THOR_RELEASE_PYTHON_EXECUTABLE OR THOR_RELEASE_PYTHON_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR
            "RepairThorWheels.cmake requires -DTHOR_RELEASE_PYTHON_EXECUTABLE=/path/to/python")
endif ()
get_filename_component(THOR_RELEASE_PYTHON_EXECUTABLE "${THOR_RELEASE_PYTHON_EXECUTABLE}" ABSOLUTE)

function(_thor_require_python_module module_name)
    execute_process(
            COMMAND "${THOR_RELEASE_PYTHON_EXECUTABLE}" -c "import ${module_name}"
            RESULT_VARIABLE _result
            OUTPUT_QUIET
            ERROR_VARIABLE _stderr
    )
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR
                "Thor release Python ${THOR_RELEASE_PYTHON_EXECUTABLE} cannot import ${module_name}.\n"
                "Install it into that interpreter before continuing.\n${_stderr}")
    endif ()
endfunction()

_thor_require_python_module(auditwheel)
_thor_require_python_module(twine)

set(_thor_external_cuda_libraries
        libcublas.so.13
        libcublasLt.so.13
        libnvJitLink.so.13
        libnvrtc.so.13
        libcudart.so.13
        libcudnn.so.9
        libcusolver.so.12
        libcusparse.so.12
)

function(_thor_repair_wheel wheel label exclude_libthor)
    set(_args repair --plat "${THOR_MANYLINUX_PLATFORM}")
    foreach (_library IN LISTS _thor_external_cuda_libraries)
        list(APPEND _args --exclude "${_library}")
    endforeach ()
    if (exclude_libthor)
        # _thor retains DT_NEEDED: libThor.so by design. Runtime bootstrap loads
        # the exact-version backend supplied by thor-cuda-kernels-smXX before
        # importing _thor, so auditwheel must not try to vendor or rename it.
        list(APPEND _args --exclude libThor.so)
    endif ()
    list(APPEND _args -w "${THOR_WHEELHOUSE_DIR}" "${wheel}")

    message(STATUS "Thor: auditwheel repair ${label}")
    execute_process(
            COMMAND "${THOR_RELEASE_PYTHON_EXECUTABLE}" -m auditwheel ${_args}
            RESULT_VARIABLE _result
            OUTPUT_VARIABLE _stdout
            ERROR_VARIABLE _stderr
    )
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR
                "auditwheel repair failed for ${label}.\n"
                "stdout:\n${_stdout}\n"
                "stderr:\n${_stderr}")
    endif ()
    if (NOT _stdout STREQUAL "")
        string(STRIP "${_stdout}" _stdout)
        message(STATUS "${_stdout}")
    endif ()
endfunction()

# A release wheelhouse is a derived artifact. Recreate it atomically from the
# three raw wheels so stale releases or duplicate repair outputs cannot be
# uploaded accidentally.
file(REMOVE_RECURSE "${THOR_WHEELHOUSE_DIR}")
file(MAKE_DIRECTORY "${THOR_WHEELHOUSE_DIR}")

_thor_repair_wheel("${_thor_sm89_wheel}" "thor-cuda-kernels-sm89" FALSE)
_thor_repair_wheel("${_thor_sm120_wheel}" "thor-cuda-kernels-sm120" FALSE)
_thor_repair_wheel("${_thor_core_wheel}" "thor-cuda" TRUE)

execute_process(
        COMMAND "${CMAKE_COMMAND}"
        "-DTHOR_WHEEL_DIR=${THOR_WHEELHOUSE_DIR}"
        "-DTHOR_EXPECT_PLATFORM_TAG=${THOR_MANYLINUX_PLATFORM}"
        "-DTHOR_PYPI_MAX_WHEEL_BYTES=${THOR_PYPI_MAX_WHEEL_BYTES}"
        -P "${CMAKE_CURRENT_LIST_DIR}/VerifyThorWheels.cmake"
        RESULT_VARIABLE _verify_result
)
if (NOT _verify_result EQUAL 0)
    message(FATAL_ERROR "Thor repaired-wheel verification failed")
endif ()

file(GLOB _thor_repaired_wheels LIST_DIRECTORIES FALSE "${THOR_WHEELHOUSE_DIR}/*.whl")
list(LENGTH _thor_repaired_wheels _thor_repaired_count)
if (NOT _thor_repaired_count EQUAL 3)
    message(FATAL_ERROR "Expected exactly three repaired Thor wheels, found ${_thor_repaired_count}: ${_thor_repaired_wheels}")
endif ()

execute_process(
        COMMAND "${THOR_RELEASE_PYTHON_EXECUTABLE}" -m twine check ${_thor_repaired_wheels}
        RESULT_VARIABLE _twine_result
        OUTPUT_VARIABLE _twine_stdout
        ERROR_VARIABLE _twine_stderr
)
if (NOT _twine_result EQUAL 0)
    message(FATAL_ERROR
            "twine check failed for repaired Thor wheels.\n"
            "stdout:\n${_twine_stdout}\n"
            "stderr:\n${_twine_stderr}")
endif ()
if (NOT _twine_stdout STREQUAL "")
    string(STRIP "${_twine_stdout}" _twine_stdout)
    message(STATUS "${_twine_stdout}")
endif ()

message(STATUS "Thor: repaired release wheel set is ready in ${THOR_WHEELHOUSE_DIR}")
