cmake_minimum_required(VERSION 3.22)

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
project(Thor VERSION ${THOR_VERSION_NUMERIC} LANGUAGES CXX CUDA)


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
# Global settings
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
#set(CMAKE_CUDA_ARCHITECTURES 75 120)

# Default to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_options(-UNDEBUG)

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
# Automatic Git versioning
# ============================================================
find_package(Git QUIET)
if (Git_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else ()
    set(GIT_VERSION "unknown")
endif ()

message(STATUS "Git version: ${GIT_VERSION}")

configure_file(
        ${CMAKE_SOURCE_DIR}/ThorVersion.h.in
        ${CMAKE_BINARY_DIR}/generated/ThorVersion.h
        @ONLY
)

# ============================================================
# Source files
# ============================================================
file(GLOB_RECURSE THOR_SOURCES
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
        CUDA_STANDARD 17
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

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(Thor PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)
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
# GraphicsMagick++ (Magick++)
# ============================================================
find_package(PkgConfig REQUIRED)
pkg_check_modules(GMAGICK REQUIRED GraphicsMagick++)

execute_process(
        COMMAND GraphicsMagick++-config --cppflags
        OUTPUT_VARIABLE GM_CPPFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND GraphicsMagick++-config --cxxflags
        OUTPUT_VARIABLE GM_CXXFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND GraphicsMagick++-config --ldflags --libs
        OUTPUT_VARIABLE GM_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add include dirs from cppflags
separate_arguments(GM_CPPFLAGS_LIST NATIVE_COMMAND "${GM_CPPFLAGS}")
separate_arguments(GM_CXXFLAGS_LIST NATIVE_COMMAND "${GM_CXXFLAGS}")
separate_arguments(GM_LDFLAGS_LIST NATIVE_COMMAND "${GM_LDFLAGS}")

# GraphicsMagick++-config can leak distro package build flags into consumers.
# In Debug builds only, strip LTO-related flags because they slow compiles/relinks
# and are not useful for normal debugging. Non-Debug builds keep the original flags.
set(GM_CXXFLAGS_DEBUG_LIST "${GM_CXXFLAGS_LIST}")
set(GM_LDFLAGS_DEBUG_LIST "${GM_LDFLAGS_LIST}")
foreach (_gm_flags_var IN ITEMS GM_CXXFLAGS_DEBUG_LIST GM_LDFLAGS_DEBUG_LIST)
    list(FILTER ${_gm_flags_var} EXCLUDE REGEX "^-flto(=.*)?$")
    list(FILTER ${_gm_flags_var} EXCLUDE REGEX "^-ffat-lto-objects$")
endforeach ()

# Build per-flag generator expressions. This avoids putting semicolon-separated
# CMake lists inside one generator expression, which can mis-handle multi-flag lists.
set(_thor_gm_cxxflags_genex "")
foreach (_gm_flag IN LISTS GM_CXXFLAGS_DEBUG_LIST)
    list(APPEND _thor_gm_cxxflags_genex
            "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${_gm_flag}>")
endforeach ()
foreach (_gm_flag IN LISTS GM_CXXFLAGS_LIST)
    list(APPEND _thor_gm_cxxflags_genex
            "$<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:${_gm_flag}>")
endforeach ()

set(_thor_gm_ldflags_genex "")
foreach (_gm_flag IN LISTS GM_LDFLAGS_DEBUG_LIST)
    list(APPEND _thor_gm_ldflags_genex "$<$<CONFIG:Debug>:${_gm_flag}>")
endforeach ()
foreach (_gm_flag IN LISTS GM_LDFLAGS_LIST)
    list(APPEND _thor_gm_ldflags_genex "$<$<NOT:$<CONFIG:Debug>>:${_gm_flag}>")
endforeach ()

target_include_directories(Thor PUBLIC ${GMAGICK_INCLUDE_DIRS})
target_compile_options(Thor PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${GM_CPPFLAGS_LIST}>
        ${_thor_gm_cxxflags_genex}
)
target_link_libraries(Thor PRIVATE ${_thor_gm_ldflags_genex})

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
        $<$<CONFIG:Debug>:THOR_DEBUG GDK_NVDIRECT _FILE_OFFSET_BITS=64 _GLIBCXX_DEBUG>
)

# CUDA-specific extra flags
target_compile_options(Thor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings>
)

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
target_link_libraries(thor_tests PRIVATE Thor GTest::gtest_main CUDA::cudart CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver)
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
target_link_libraries(thor_embedding_sparse_backward_profile PRIVATE Thor CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver)
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
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/headerlist.txt
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND find . -name "*.h" | grep -v ./googletest/ | grep -v ./googlemock/ | grep -v ./_deps/ | grep -v ./build/ | grep -v ./cmake-build- | grep -v ./out/ | grep -v ./_build/ | grep -v ./generated/ | grep -v ./test/ | grep -v ./.venv/ | grep -v ./bindings > ${CMAKE_BINARY_DIR}/headerlist.txt
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${THOR_SOURCES}
        COMMENT "Collecting all headers into headerlist.txt"
)

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

# Compute GIT_VERSION at configure time
execute_process(
        COMMAND git describe --always --dirty --tags
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(${CMAKE_SOURCE_DIR}/ThorVersion.h.in
        ${CMAKE_BINARY_DIR}/generated/ThorVersion.h
        @ONLY)

# Reconfigure when HEAD changes (and when refs/index change)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/.git/HEAD"
        "${CMAKE_SOURCE_DIR}/.git/index"
        "${CMAKE_SOURCE_DIR}/.git/packed-refs"
)

# Also depend on the currently checked-out branch ref, if available
file(READ "${CMAKE_SOURCE_DIR}/.git/HEAD" _git_head LIMIT 200)
if (_git_head MATCHES "^ref: (.+)$")
    set(_git_ref "${CMAKE_SOURCE_DIR}/.git/${CMAKE_MATCH_1}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_git_ref}")
endif ()

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
