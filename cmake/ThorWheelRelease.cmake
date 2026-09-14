cmake_minimum_required(VERSION 3.28)

# Stable source-tree entry point for post-PEP517 Thor wheel release operations.
# The scikit-build-core CMake tree records build tools from its isolated PEP 517
# environment and therefore must not be reused after `python -m build --wheel`
# exits.  This script operates directly on the completed wheel artifacts.

if (NOT DEFINED THOR_RELEASE_ACTION OR THOR_RELEASE_ACTION STREQUAL "")
    message(FATAL_ERROR
            "ThorWheelRelease.cmake requires -DTHOR_RELEASE_ACTION=repair|verify|publish")
endif ()
string(TOLOWER "${THOR_RELEASE_ACTION}" THOR_RELEASE_ACTION)
if (NOT THOR_RELEASE_ACTION MATCHES "^(repair|verify|publish)$")
    message(FATAL_ERROR
            "Unsupported THOR_RELEASE_ACTION='${THOR_RELEASE_ACTION}'. Expected repair, verify, or publish.")
endif ()

get_filename_component(_thor_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Post-build release tools (auditwheel and Twine) are installed into the
# release Python environment.  Invoke them as modules instead of relying on
# console-script wrappers being present on PATH.
if (NOT DEFINED THOR_RELEASE_PYTHON_EXECUTABLE OR THOR_RELEASE_PYTHON_EXECUTABLE STREQUAL "")
    if (DEFINED ENV{THOR_RELEASE_PYTHON_EXECUTABLE} AND NOT "$ENV{THOR_RELEASE_PYTHON_EXECUTABLE}" STREQUAL "")
        set(THOR_RELEASE_PYTHON_EXECUTABLE "$ENV{THOR_RELEASE_PYTHON_EXECUTABLE}")
    elseif (EXISTS "/opt/python/cp312-cp312/bin/python")
        set(THOR_RELEASE_PYTHON_EXECUTABLE "/opt/python/cp312-cp312/bin/python")
    else ()
        find_program(THOR_RELEASE_PYTHON_EXECUTABLE NAMES python3 python REQUIRED)
    endif ()
endif ()
get_filename_component(THOR_RELEASE_PYTHON_EXECUTABLE "${THOR_RELEASE_PYTHON_EXECUTABLE}" ABSOLUTE)
if (NOT EXISTS "${THOR_RELEASE_PYTHON_EXECUTABLE}")
    message(FATAL_ERROR
            "Thor release Python does not exist: ${THOR_RELEASE_PYTHON_EXECUTABLE}. "
            "Pass -DTHOR_RELEASE_PYTHON_EXECUTABLE=/path/to/python.")
endif ()
message(STATUS "Thor: release Python = ${THOR_RELEASE_PYTHON_EXECUTABLE}")

if (NOT DEFINED THOR_WHEEL_DIST_DIR OR THOR_WHEEL_DIST_DIR STREQUAL "")
    set(THOR_WHEEL_DIST_DIR "${_thor_repo_root}/bindings/python/dist")
endif ()
if (NOT DEFINED THOR_WHEELHOUSE_DIR OR THOR_WHEELHOUSE_DIR STREQUAL "")
    set(THOR_WHEELHOUSE_DIR "${_thor_repo_root}/bindings/python/wheelhouse")
endif ()
if (NOT DEFINED THOR_MANYLINUX_PLATFORM OR THOR_MANYLINUX_PLATFORM STREQUAL "")
    set(THOR_MANYLINUX_PLATFORM "manylinux_2_28_x86_64")
endif ()
if (NOT DEFINED THOR_PYPI_MAX_WHEEL_BYTES OR THOR_PYPI_MAX_WHEEL_BYTES STREQUAL "")
    set(THOR_PYPI_MAX_WHEEL_BYTES 104857600)
endif ()

get_filename_component(THOR_WHEEL_DIST_DIR "${THOR_WHEEL_DIST_DIR}" ABSOLUTE)
get_filename_component(THOR_WHEELHOUSE_DIR "${THOR_WHEELHOUSE_DIR}" ABSOLUTE)

set(_thor_repair_script "${CMAKE_CURRENT_LIST_DIR}/RepairThorWheels.cmake")
set(_thor_verify_script "${CMAKE_CURRENT_LIST_DIR}/VerifyThorWheels.cmake")
set(_thor_publish_script "${CMAKE_CURRENT_LIST_DIR}/PublishThorWheels.cmake")

function(_thor_run_cmake_script script label)
    set(_args ${ARGN})
    execute_process(
            COMMAND "${CMAKE_COMMAND}" ${_args} -P "${script}"
            RESULT_VARIABLE _result
    )
    if (NOT _result EQUAL 0)
        message(FATAL_ERROR "Thor ${label} failed")
    endif ()
endfunction()

function(_thor_repair_release_set)
    message(STATUS "Thor: repairing raw wheel set from ${THOR_WHEEL_DIST_DIR}")
    _thor_run_cmake_script(
            "${_thor_repair_script}"
            "wheel repair"
            "-DTHOR_WHEEL_DIST_DIR=${THOR_WHEEL_DIST_DIR}"
            "-DTHOR_WHEELHOUSE_DIR=${THOR_WHEELHOUSE_DIR}"
            "-DTHOR_MANYLINUX_PLATFORM=${THOR_MANYLINUX_PLATFORM}"
            "-DTHOR_PYPI_MAX_WHEEL_BYTES=${THOR_PYPI_MAX_WHEEL_BYTES}"
            "-DTHOR_RELEASE_PYTHON_EXECUTABLE=${THOR_RELEASE_PYTHON_EXECUTABLE}"
    )
endfunction()

if (THOR_RELEASE_ACTION STREQUAL "repair")
    _thor_repair_release_set()
    message(STATUS "Thor: repaired release wheel set is ready in ${THOR_WHEELHOUSE_DIR}")
elseif (THOR_RELEASE_ACTION STREQUAL "verify")
    message(STATUS "Thor: verifying repaired wheel set in ${THOR_WHEELHOUSE_DIR}")
    _thor_run_cmake_script(
            "${_thor_verify_script}"
            "wheel verification"
            "-DTHOR_WHEEL_DIR=${THOR_WHEELHOUSE_DIR}"
            "-DTHOR_EXPECT_PLATFORM_TAG=${THOR_MANYLINUX_PLATFORM}"
            "-DTHOR_PYPI_MAX_WHEEL_BYTES=${THOR_PYPI_MAX_WHEEL_BYTES}"
    )
elseif (THOR_RELEASE_ACTION STREQUAL "publish")
    # Preserve the old target dependency: publication always rebuilds the
    # wheelhouse from the raw dist/ wheels and passes all release gates first.
    _thor_repair_release_set()

    message(STATUS "Thor: publishing verified wheel set from ${THOR_WHEELHOUSE_DIR}")
    set(_publish_args
            "-DTHOR_WHEELHOUSE_DIR=${THOR_WHEELHOUSE_DIR}"
            "-DTHOR_MANYLINUX_PLATFORM=${THOR_MANYLINUX_PLATFORM}"
            "-DTHOR_RELEASE_PYTHON_EXECUTABLE=${THOR_RELEASE_PYTHON_EXECUTABLE}"
    )
    if (DEFINED THOR_PUBLISH_CONFIRM AND NOT THOR_PUBLISH_CONFIRM STREQUAL "")
        list(APPEND _publish_args "-DTHOR_PUBLISH_CONFIRM=${THOR_PUBLISH_CONFIRM}")
    endif ()
    if (DEFINED THOR_TWINE_REPOSITORY AND NOT THOR_TWINE_REPOSITORY STREQUAL "")
        list(APPEND _publish_args "-DTHOR_TWINE_REPOSITORY=${THOR_TWINE_REPOSITORY}")
    endif ()
    if (DEFINED THOR_TWINE_REPOSITORY_URL AND NOT THOR_TWINE_REPOSITORY_URL STREQUAL "")
        list(APPEND _publish_args "-DTHOR_TWINE_REPOSITORY_URL=${THOR_TWINE_REPOSITORY_URL}")
    endif ()

    _thor_run_cmake_script(
            "${_thor_publish_script}"
            "wheel publication"
            ${_publish_args}
    )
endif ()
