cmake_minimum_required(VERSION 3.28)

foreach (_required IN ITEMS
        THOR_KERNEL_LIBRARY
        THOR_KERNEL_SM
        THOR_KERNEL_DISTRIBUTION
        THOR_KERNEL_VERSION
        THOR_KERNEL_DESCRIPTION
        THOR_WHEEL_PLATFORM_TAG
        THOR_WHEEL_OUTPUT_DIR
        THOR_WHEEL_STAGE_ROOT
        THOR_PYTHON_EXECUTABLE
        THOR_LICENSE_FILE
)
    if (NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "PackThorKernelWheel.cmake requires -D${_required}=...")
    endif ()
endforeach ()

if (NOT EXISTS "${THOR_KERNEL_LIBRARY}")
    message(FATAL_ERROR "Thor kernel backend library does not exist: ${THOR_KERNEL_LIBRARY}")
endif ()
if (NOT EXISTS "${THOR_LICENSE_FILE}")
    message(FATAL_ERROR "Thor license file does not exist: ${THOR_LICENSE_FILE}")
endif ()

string(REPLACE "-" "_" _distribution_normalized "${THOR_KERNEL_DISTRIBUTION}")
set(_wheel_tag "py3-none-${THOR_WHEEL_PLATFORM_TAG}")
set(_stage_root "${THOR_WHEEL_STAGE_ROOT}/${_distribution_normalized}-${THOR_KERNEL_VERSION}")
set(_package_dir "${_stage_root}/thor_cuda_kernels/sm${THOR_KERNEL_SM}")
set(_dist_info_dir "${_stage_root}/${_distribution_normalized}-${THOR_KERNEL_VERSION}.dist-info")

file(REMOVE_RECURSE "${_stage_root}")
file(MAKE_DIRECTORY "${_package_dir}" "${_dist_info_dir}/licenses" "${THOR_WHEEL_OUTPUT_DIR}")
file(COPY_FILE "${THOR_KERNEL_LIBRARY}" "${_package_dir}/libThor.so" ONLY_IF_DIFFERENT)
file(COPY_FILE "${THOR_LICENSE_FILE}" "${_dist_info_dir}/licenses/LICENSE" ONLY_IF_DIFFERENT)

file(WRITE "${_dist_info_dir}/METADATA"
"Metadata-Version: 2.4\n"
"Name: ${THOR_KERNEL_DISTRIBUTION}\n"
"Version: ${THOR_KERNEL_VERSION}\n"
"Summary: ${THOR_KERNEL_DESCRIPTION}\n"
"Requires-Python: >=3.12\n"
"License-Expression: Apache-2.0\n"
"License-File: LICENSE\n"
"Description-Content-Type: text/markdown\n"
"\n"
"# ${THOR_KERNEL_DISTRIBUTION}\n"
"\n"
"${THOR_KERNEL_DESCRIPTION}. This package is installed automatically by `thor-cuda` and is not intended to be used directly.\n"
)

file(WRITE "${_dist_info_dir}/WHEEL"
"Wheel-Version: 1.0\n"
"Generator: Thor CMake release-wheel bundle\n"
"Root-Is-Purelib: false\n"
"Tag: ${_wheel_tag}\n"
"\n"
)

execute_process(
        COMMAND "${THOR_PYTHON_EXECUTABLE}" -m wheel pack "${_stage_root}" -d "${THOR_WHEEL_OUTPUT_DIR}"
        RESULT_VARIABLE _wheel_pack_result
        OUTPUT_VARIABLE _wheel_pack_output
        ERROR_VARIABLE _wheel_pack_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
)
if (NOT _wheel_pack_result EQUAL 0)
    message(FATAL_ERROR
            "Failed to package ${THOR_KERNEL_DISTRIBUTION} from ${THOR_KERNEL_LIBRARY}.\n"
            "stdout:\n${_wheel_pack_output}\n"
            "stderr:\n${_wheel_pack_error}")
endif ()

message(STATUS "Thor: packaged ${THOR_KERNEL_DISTRIBUTION} ${THOR_KERNEL_VERSION} (${_wheel_tag}) into ${THOR_WHEEL_OUTPUT_DIR}")
