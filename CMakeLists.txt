cmake_minimum_required(VERSION 3.22)

# Read THOR_VERSION from ThorVersion.h.in
file(READ "${CMAKE_SOURCE_DIR}/ThorVersion.h.in" _thor_version_header)

# 1) Full version string, e.g. "0.1.1.dev0"
string(
        REGEX MATCH
        "#define[ \t]+THOR_VERSION[ \t]+\"([^\"]*)\""
        _thor_version_match
        "${_thor_version_header}"
)

if(NOT _thor_version_match)
    message(FATAL_ERROR "Could not find THOR_VERSION in ThorVersion.h.in")
endif()

set(THOR_VERSION_FULL "${CMAKE_MATCH_1}")  # e.g. "0.1.1.dev0"

# 2) Numeric prefix for CMake project version, e.g. "0.1.1"
string(
        REGEX MATCH
        "^[0-9]+(\\.[0-9]+)*"
        _thor_version_numeric_match
        "${THOR_VERSION_FULL}"
)

if(NOT _thor_version_numeric_match)
    message(FATAL_ERROR "THOR_VERSION must start with numeric semver, got ${THOR_VERSION_FULL}")
endif()

set(THOR_VERSION_NUMERIC "${CMAKE_MATCH_0}")  # "0.1.1"

# 3) Use numeric for CMake's project version
project(Thor VERSION ${THOR_VERSION_NUMERIC} LANGUAGES CXX CUDA)

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
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
#set(CMAKE_CUDA_ARCHITECTURES 75 120)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_options(-UNDEBUG)

# ============================================================
# Automatic Git versioning
# ============================================================
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    set(GIT_VERSION "unknown")
endif()

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
        -Wl,--no-as-needed
        )

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(Thor PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(Thor PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cusolver
        )

# Manually include cuDNN and cuFile
target_link_directories(Thor PRIVATE
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        )

target_link_libraries(Thor PRIVATE
        cudnn
        cufile
        )

target_include_directories(Thor PUBLIC
        $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
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

target_include_directories(Thor PUBLIC ${GMAGICK_INCLUDE_DIRS})
target_compile_options(Thor PRIVATE ${GM_CPPFLAGS_LIST} ${GM_CXXFLAGS_LIST})
target_link_libraries(Thor PRIVATE ${GM_LDFLAGS_LIST})

# ============================================================
# Boost
# ============================================================
find_package(Boost REQUIRED)
target_link_libraries(Thor PRIVATE Boost::boost)

# Per-configuration flags
target_compile_options(Thor PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3 -ggdb -fsanitize=undefined>
        $<$<CONFIG:Debug>:-O0 -ggdb>
        )

target_compile_definitions(Thor PRIVATE
        $<$<CONFIG:Release>:THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64>
        $<$<CONFIG:RelWithDebInfo>:THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64>
        $<$<CONFIG:Debug>:THOR_DEBUG GDK_NVDIRECT _FILE_OFFSET_BITS=64 _GLIBCXX_DEBUG>
        )

# CUDA-specific extra flags
target_compile_options(Thor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler --Werror all-warnings -Xcompiler -fPIC>
        )

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

file(GLOB_RECURSE TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp")
add_executable(thor_tests ${TEST_SOURCES})
target_link_libraries(thor_tests PRIVATE Thor GTest::gtest_main CUDA::cudart CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver)
# Manually include cuDNN and cuFile
target_link_directories(thor_tests PRIVATE
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        )
target_link_libraries(thor_tests PRIVATE
        cudnn
        cufile
        )
find_package(OpenMP REQUIRED)
target_link_libraries(thor_tests PRIVATE OpenMP::OpenMP_CXX)
find_package(Boost REQUIRED)
target_link_libraries(thor_tests PRIVATE Boost::boost)
include(GoogleTest)
gtest_discover_tests(thor_tests)

# Per-configuration flags
target_compile_definitions(thor_tests PRIVATE
        $<$<CONFIG:Release>:THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64 SOURCE_DIR="${CMAKE_SOURCE_DIR}">
        $<$<CONFIG:RelWithDebInfo>:THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64 SOURCE_DIR="${CMAKE_SOURCE_DIR}">
        $<$<CONFIG:Debug>:THOR_DEBUG GDK_NVDIRECT _FILE_OFFSET_BITS=64 _GLIBCXX_DEBUG SOURCE_DIR="${CMAKE_SOURCE_DIR}">
        )

target_compile_options(thor_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:Debug>:-O0 -ggdb>
        )

# ============================================================
# Install & Export for find_package()
# ============================================================
install(TARGETS Thor
        EXPORT ThorTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
install(FILES Thor.h ${CMAKE_BINARY_DIR}/generated/ThorVersion.h DESTINATION include)

install(EXPORT ThorTargets
        FILE ThorTargets.cmake
        NAMESPACE Thor::
        DESTINATION lib/cmake/Thor
        )

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

install(FILES
        ${CMAKE_BINARY_DIR}/ThorConfig.cmake
        ${CMAKE_BINARY_DIR}/ThorConfigVersion.cmake
        DESTINATION lib/cmake/Thor
        )


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
        COMMAND find . -name "*.h" | grep -v ./googletest/ | grep -v ./googlemock/ | grep -v ./_deps/ | grep -v ./build/ | grep -v ./test/ | grep -v ./.venv/ | grep -v ./bindings > ${CMAKE_BINARY_DIR}/headerlist.txt
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

# Python bindings
option(THOR_USE_PROJECT_VENV "Prefer the project's .venv for Python" ON)

if(THOR_USE_PROJECT_VENV)
    set(_thor_venv_python "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
    message(STATUS "Thor: using project venv Python at ${_thor_venv_python}")

    # Clear old cached Python detection (3.9, etc.)
    unset(Python3_EXECUTABLE CACHE)
    unset(Python3_LIBRARY CACHE)
    unset(Python3_INCLUDE_DIR CACHE)
    unset(Python_EXECUTABLE CACHE)
    unset(Python_LIBRARY CACHE)
    unset(Python_INCLUDE_DIR CACHE)

    set(Python3_EXECUTABLE "${_thor_venv_python}" CACHE FILEPATH "Python3 executable for Thor bindings" FORCE)
    set(Python_EXECUTABLE  "${_thor_venv_python}" CACHE FILEPATH "Python executable for Thor bindings (nanobind)" FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

option(THOR_BUILD_PYTHON "Build Python bindings" ON)
if(THOR_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()

# For convenience, rebuild the python bindings before rebuilding the c++ tests
add_dependencies(thor_tests thor)