cmake_minimum_required(VERSION 3.28)

if (NOT DEFINED THOR_WHEELHOUSE_DIR OR THOR_WHEELHOUSE_DIR STREQUAL "")
    message(FATAL_ERROR "PublishThorWheels.cmake requires -DTHOR_WHEELHOUSE_DIR=/path/to/wheelhouse")
endif ()
if (NOT DEFINED THOR_MANYLINUX_PLATFORM OR THOR_MANYLINUX_PLATFORM STREQUAL "")
    set(THOR_MANYLINUX_PLATFORM "manylinux_2_28_x86_64")
endif ()

if (NOT DEFINED THOR_PUBLISH_CONFIRM OR THOR_PUBLISH_CONFIRM STREQUAL "")
    set(THOR_PUBLISH_CONFIRM "$ENV{THOR_PUBLISH_CONFIRM}")
endif ()
if (NOT THOR_PUBLISH_CONFIRM STREQUAL "YES")
    message(FATAL_ERROR
            "Publishing is intentionally gated. Set THOR_PUBLISH_CONFIRM=YES in the environment "
            "or pass -DTHOR_PUBLISH_CONFIRM=YES after verifying the wheelhouse.")
endif ()

if (NOT DEFINED THOR_TWINE_REPOSITORY OR THOR_TWINE_REPOSITORY STREQUAL "")
    set(THOR_TWINE_REPOSITORY "$ENV{THOR_TWINE_REPOSITORY}")
endif ()
if (NOT DEFINED THOR_TWINE_REPOSITORY_URL OR THOR_TWINE_REPOSITORY_URL STREQUAL "")
    set(THOR_TWINE_REPOSITORY_URL "$ENV{THOR_TWINE_REPOSITORY_URL}")
endif ()
if (NOT THOR_TWINE_REPOSITORY STREQUAL "" AND NOT THOR_TWINE_REPOSITORY_URL STREQUAL "")
    message(FATAL_ERROR "Set at most one of THOR_TWINE_REPOSITORY and THOR_TWINE_REPOSITORY_URL")
endif ()

get_filename_component(THOR_WHEELHOUSE_DIR "${THOR_WHEELHOUSE_DIR}" ABSOLUTE)
get_filename_component(_thor_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(READ "${_thor_repo_root}/ThorVersion.h.in" _thor_version_text)
string(REGEX MATCH "#define[ \t]+THOR_VERSION[ \t]+\"v?([^\"]+)\"" _thor_version_match "${_thor_version_text}")
if (NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Unable to read THOR_VERSION from ${_thor_repo_root}/ThorVersion.h.in")
endif ()
set(_thor_version "${CMAKE_MATCH_1}")

execute_process(
        COMMAND "${CMAKE_COMMAND}"
        "-DTHOR_WHEEL_DIR=${THOR_WHEELHOUSE_DIR}"
        "-DTHOR_EXPECT_PLATFORM_TAG=${THOR_MANYLINUX_PLATFORM}"
        -P "${CMAKE_CURRENT_LIST_DIR}/VerifyThorWheels.cmake"
        RESULT_VARIABLE _verify_result
)
if (NOT _verify_result EQUAL 0)
    message(FATAL_ERROR "Refusing to publish: Thor wheel verification failed")
endif ()

function(_thor_require_publish_wheel pattern label out_var)
    file(GLOB _matches LIST_DIRECTORIES FALSE "${pattern}")
    list(LENGTH _matches _count)
    if (NOT _count EQUAL 1)
        message(FATAL_ERROR "Expected exactly one ${label} wheel to publish, found ${_count}: ${_matches}")
    endif ()
    list(GET _matches 0 _wheel)
    set(${out_var} "${_wheel}" PARENT_SCOPE)
endfunction()

_thor_require_publish_wheel(
        "${THOR_WHEELHOUSE_DIR}/thor_cuda_kernels_sm89-${_thor_version}-*.whl"
        "thor-cuda-kernels-sm89"
        _thor_sm89_wheel
)
_thor_require_publish_wheel(
        "${THOR_WHEELHOUSE_DIR}/thor_cuda_kernels_sm120-${_thor_version}-*.whl"
        "thor-cuda-kernels-sm120"
        _thor_sm120_wheel
)
_thor_require_publish_wheel(
        "${THOR_WHEELHOUSE_DIR}/thor_cuda-${_thor_version}-*.whl"
        "thor-cuda"
        _thor_core_wheel
)

if (NOT DEFINED THOR_RELEASE_PYTHON_EXECUTABLE OR THOR_RELEASE_PYTHON_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR
            "PublishThorWheels.cmake requires -DTHOR_RELEASE_PYTHON_EXECUTABLE=/path/to/python")
endif ()
get_filename_component(THOR_RELEASE_PYTHON_EXECUTABLE "${THOR_RELEASE_PYTHON_EXECUTABLE}" ABSOLUTE)
execute_process(
        COMMAND "${THOR_RELEASE_PYTHON_EXECUTABLE}" -c "import twine"
        RESULT_VARIABLE _twine_import_result
        OUTPUT_QUIET
        ERROR_VARIABLE _twine_import_stderr
)
if (NOT _twine_import_result EQUAL 0)
    message(FATAL_ERROR
            "Thor release Python ${THOR_RELEASE_PYTHON_EXECUTABLE} cannot import twine.\n"
            "Install Twine into that interpreter before continuing.\n${_twine_import_stderr}")
endif ()

set(_twine_destination_args "")
if (NOT THOR_TWINE_REPOSITORY STREQUAL "")
    list(APPEND _twine_destination_args --repository "${THOR_TWINE_REPOSITORY}")
elseif (NOT THOR_TWINE_REPOSITORY_URL STREQUAL "")
    list(APPEND _twine_destination_args --repository-url "${THOR_TWINE_REPOSITORY_URL}")
endif ()

function(_thor_publish_one wheel distribution_name)
    message(STATUS "Thor: publishing ${distribution_name}: ${wheel}")
    execute_process(
            COMMAND "${THOR_RELEASE_PYTHON_EXECUTABLE}" -m twine upload ${_twine_destination_args} "${wheel}"
            RESULT_VARIABLE _result
            OUTPUT_VARIABLE _stdout
            ERROR_VARIABLE _stderr
    )
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR
                "twine upload failed for ${distribution_name}; later Thor distributions were not uploaded.\n"
                "stdout:\n${_stdout}\n"
                "stderr:\n${_stderr}")
    endif ()
    if (NOT _stdout STREQUAL "")
        string(STRIP "${_stdout}" _stdout)
        message(STATUS "${_stdout}")
    endif ()
endfunction()

# Dependency-first publication is intentional. thor-cuda requires both kernel
# distributions at the exact same version, so the public package must be the
# final upload in the release sequence.
_thor_publish_one("${_thor_sm89_wheel}" "thor-cuda-kernels-sm89")
_thor_publish_one("${_thor_sm120_wheel}" "thor-cuda-kernels-sm120")
_thor_publish_one("${_thor_core_wheel}" "thor-cuda")

message(STATUS "Thor: publication completed in dependency order (sm89, sm120, thor-cuda)")
