cmake_minimum_required(VERSION 3.28)

if (NOT DEFINED THOR_WHEEL_DIR OR THOR_WHEEL_DIR STREQUAL "")
    message(FATAL_ERROR "VerifyThorWheels.cmake requires -DTHOR_WHEEL_DIR=/path/to/wheelhouse")
endif ()

if (NOT DEFINED THOR_EXPECT_PLATFORM_TAG OR THOR_EXPECT_PLATFORM_TAG STREQUAL "")
    set(THOR_EXPECT_PLATFORM_TAG "manylinux_2_28_x86_64")
endif ()
if (NOT DEFINED THOR_PYPI_MAX_WHEEL_BYTES OR THOR_PYPI_MAX_WHEEL_BYTES STREQUAL "")
    set(THOR_PYPI_MAX_WHEEL_BYTES 104857600)
endif ()

get_filename_component(THOR_WHEEL_DIR "${THOR_WHEEL_DIR}" ABSOLUTE)
get_filename_component(_thor_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_thor_version_header "${_thor_repo_root}/ThorVersion.h.in")
if (NOT EXISTS "${_thor_version_header}")
    message(FATAL_ERROR "Thor version header not found: ${_thor_version_header}")
endif ()

file(READ "${_thor_version_header}" _thor_version_text)
string(REGEX MATCH "#define[ \t]+THOR_VERSION[ \t]+\"v?([^\"]+)\"" _thor_version_match "${_thor_version_text}")
if (NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Unable to read THOR_VERSION from ${_thor_version_header}")
endif ()
set(_thor_version "${CMAKE_MATCH_1}")

function(_thor_require_single_wheel pattern label out_var)
    file(GLOB _matches LIST_DIRECTORIES FALSE "${pattern}")
    list(LENGTH _matches _count)
    if (NOT _count EQUAL 1)
        message(FATAL_ERROR
                "Expected exactly one ${label} wheel matching:\n"
                "  ${pattern}\n"
                "Found ${_count}: ${_matches}")
    endif ()
    list(GET _matches 0 _wheel)
    set(${out_var} "${_wheel}" PARENT_SCOPE)
endfunction()

function(_thor_require_metadata stage_dir expected_name)
    file(GLOB _metadata_files LIST_DIRECTORIES FALSE "${stage_dir}/*.dist-info/METADATA")
    list(LENGTH _metadata_files _metadata_count)
    if (NOT _metadata_count EQUAL 1)
        message(FATAL_ERROR "${expected_name}: expected exactly one .dist-info/METADATA, found ${_metadata_count}")
    endif ()
    list(GET _metadata_files 0 _metadata_file)
    file(READ "${_metadata_file}" _metadata)
    string(FIND "${_metadata}" "Name: ${expected_name}\n" _name_index)
    if (_name_index EQUAL -1)
        message(FATAL_ERROR "${expected_name}: wheel metadata has the wrong distribution name")
    endif ()
    string(FIND "${_metadata}" "Version: ${_thor_version}\n" _version_index)
    if (_version_index EQUAL -1)
        message(FATAL_ERROR "${expected_name}: wheel metadata is not version ${_thor_version}")
    endif ()

    if (expected_name STREQUAL "thor-cuda")
        foreach (_kernel_distribution IN ITEMS thor-cuda-kernels-sm89 thor-cuda-kernels-sm120)
            string(FIND
                    "${_metadata}"
                    "Requires-Dist: ${_kernel_distribution}==${_thor_version}\n"
                    _dependency_index
            )
            if (_dependency_index EQUAL -1)
                message(FATAL_ERROR
                        "thor-cuda: missing exact-version dependency ${_kernel_distribution}==${_thor_version}")
            endif ()
        endforeach ()
    endif ()

    file(GLOB _wheel_metadata_files LIST_DIRECTORIES FALSE "${stage_dir}/*.dist-info/WHEEL")
    list(LENGTH _wheel_metadata_files _wheel_metadata_count)
    if (NOT _wheel_metadata_count EQUAL 1)
        message(FATAL_ERROR "${expected_name}: expected exactly one .dist-info/WHEEL, found ${_wheel_metadata_count}")
    endif ()
    list(GET _wheel_metadata_files 0 _wheel_metadata_file)
    file(READ "${_wheel_metadata_file}" _wheel_metadata)
    string(FIND "${_wheel_metadata}" "${THOR_EXPECT_PLATFORM_TAG}" _wheel_tag_index)
    if (_wheel_tag_index EQUAL -1)
        message(FATAL_ERROR
                "${expected_name}: internal WHEEL metadata is missing platform tag ${THOR_EXPECT_PLATFORM_TAG}")
    endif ()
endfunction()

function(_thor_reject_bundled_nvidia_libraries stage_dir distribution_name)
    file(GLOB_RECURSE _shared_objects LIST_DIRECTORIES FALSE "${stage_dir}/*.so" "${stage_dir}/*.so.*")
    foreach (_shared_object IN LISTS _shared_objects)
        get_filename_component(_basename "${_shared_object}" NAME)
        if (_basename MATCHES "^(libcuda|libcudart|libcublas|libcublasLt|libnvJitLink|libnvrtc|libcudnn|libcusolver|libcusparse).*\\.so")
            message(FATAL_ERROR
                    "${distribution_name}: NVIDIA runtime library was bundled into the wheel: ${_basename}\n"
                    "Thor's CUDA user-space libraries must remain external Python dependencies.")
        endif ()
    endforeach ()
endfunction()

function(_thor_verify_wheel wheel distribution_name expected_backend_sm)
    get_filename_component(_wheel_name "${wheel}" NAME)
    string(FIND "${_wheel_name}" "${THOR_EXPECT_PLATFORM_TAG}" _platform_index)
    if (_platform_index EQUAL -1)
        message(FATAL_ERROR
                "${distribution_name}: repaired wheel is missing platform tag ${THOR_EXPECT_PLATFORM_TAG}: ${_wheel_name}")
    endif ()

    file(SIZE "${wheel}" _wheel_size)
    if (_wheel_size GREATER THOR_PYPI_MAX_WHEEL_BYTES)
        math(EXPR _wheel_mib "${_wheel_size} / 1048576")
        math(EXPR _limit_mib "${THOR_PYPI_MAX_WHEEL_BYTES} / 1048576")
        message(FATAL_ERROR
                "${distribution_name}: ${_wheel_mib} MiB wheel exceeds the configured PyPI limit of ${_limit_mib} MiB: ${wheel}")
    endif ()

    set(_stage "${THOR_WHEEL_DIR}/.thor_verify/${distribution_name}")
    file(REMOVE_RECURSE "${_stage}")
    file(MAKE_DIRECTORY "${_stage}")
    file(ARCHIVE_EXTRACT INPUT "${wheel}" DESTINATION "${_stage}")

    _thor_require_metadata("${_stage}" "${distribution_name}")
    _thor_reject_bundled_nvidia_libraries("${_stage}" "${distribution_name}")

    file(GLOB_RECURSE _thor_libraries LIST_DIRECTORIES FALSE "${_stage}/*libThor*.so" "${_stage}/*libThor*.so.*")
    if (expected_backend_sm STREQUAL "")
        if (_thor_libraries)
            message(FATAL_ERROR
                    "thor-cuda must not contain libThor.so; the native backend is supplied by the kernel wheels: ${_thor_libraries}")
        endif ()
    else ()
        set(_expected_library "${_stage}/thor_cuda_kernels/sm${expected_backend_sm}/libThor.so")
        if (NOT EXISTS "${_expected_library}")
            message(FATAL_ERROR
                    "${distribution_name}: expected backend is missing: thor_cuda_kernels/sm${expected_backend_sm}/libThor.so")
        endif ()
        list(LENGTH _thor_libraries _thor_library_count)
        if (NOT _thor_library_count EQUAL 1)
            message(FATAL_ERROR
                    "${distribution_name}: expected exactly one libThor backend, found ${_thor_library_count}: ${_thor_libraries}")
        endif ()
    endif ()

    math(EXPR _wheel_mib_tenths "(${_wheel_size} * 10) / 1048576")
    math(EXPR _wheel_mib_whole "${_wheel_mib_tenths} / 10")
    math(EXPR _wheel_mib_fraction "${_wheel_mib_tenths} % 10")
    message(STATUS
            "Thor: verified ${distribution_name} ${_thor_version}: ${_wheel_mib_whole}.${_wheel_mib_fraction} MiB (${_wheel_name})")
endfunction()

if (NOT IS_DIRECTORY "${THOR_WHEEL_DIR}")
    message(FATAL_ERROR "Thor wheel directory does not exist: ${THOR_WHEEL_DIR}")
endif ()

_thor_require_single_wheel(
        "${THOR_WHEEL_DIR}/thor_cuda-${_thor_version}-*.whl"
        "thor-cuda"
        _thor_core_wheel
)
_thor_require_single_wheel(
        "${THOR_WHEEL_DIR}/thor_cuda_kernels_sm89-${_thor_version}-*.whl"
        "thor-cuda-kernels-sm89"
        _thor_sm89_wheel
)
_thor_require_single_wheel(
        "${THOR_WHEEL_DIR}/thor_cuda_kernels_sm120-${_thor_version}-*.whl"
        "thor-cuda-kernels-sm120"
        _thor_sm120_wheel
)

_thor_verify_wheel("${_thor_core_wheel}" "thor-cuda" "")
_thor_verify_wheel("${_thor_sm89_wheel}" "thor-cuda-kernels-sm89" "89")
_thor_verify_wheel("${_thor_sm120_wheel}" "thor-cuda-kernels-sm120" "120")

file(REMOVE_RECURSE "${THOR_WHEEL_DIR}/.thor_verify")
message(STATUS "Thor: release wheel set verified successfully")
