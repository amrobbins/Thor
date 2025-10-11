cmake_minimum_required(VERSION 3.22)
project(Thor VERSION 1.0 LANGUAGES CXX CUDA)

# ============================================================
# Global settings
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -UNDEBUG")
#set(CMAKE_CUDA_ARCHITECTURES 75 120)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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
        "${CMAKE_SOURCE_DIR}/DeepLearning/**/*.cpp"
        "${CMAKE_SOURCE_DIR}/DeepLearning/**/*.cu"
        "${CMAKE_SOURCE_DIR}/Utilities/**/*.cpp"
        "${CMAKE_SOURCE_DIR}/Utilities/**/*.cu"
        "${CMAKE_SOURCE_DIR}/Demos/**/*.cpp"
        "${CMAKE_SOURCE_DIR}/Demos/**/*.cu"
        )

# ============================================================
# Create shared library
# ============================================================
add_library(Thor STATIC ${THOR_SOURCES})
#add_library(Thor SHARED ${THOR_SOURCES})
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
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings>
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


# ============================================================
# 📦 GraphicsMagick++ (Magick++)
# ============================================================
find_path(GMAGICK_INCLUDE_DIR Magick++.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES GraphicsMagick
        )

find_library(GMAGICK_LIB
        NAMES GraphicsMagick++ Magick++
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
        )

if(NOT GMAGICK_INCLUDE_DIR OR NOT GMAGICK_LIB)
    message(FATAL_ERROR "GraphicsMagick++ not found. Please install libgraphicsmagick++-dev.")
endif()

target_include_directories(Thor PUBLIC ${GMAGICK_INCLUDE_DIR})
target_link_libraries(Thor PUBLIC ${GMAGICK_LIB})


find_package(Boost REQUIRED COMPONENTS filesystem)
target_include_directories(Thor PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(Thor PRIVATE Boost::filesystem)

# Per-configuration flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(Thor PRIVATE THOR_RELEASE GDK_NVDIRECT _FILE_OFFSET_BITS=64)
    target_compile_options(Thor PRIVATE -O3)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3,-v")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(Thor PRIVATE THOR_DEBUG GDK_NVDIRECT _FILE_OFFSET_BITS=64 _GLIBCXX_DEBUG)
    target_compile_options(Thor PRIVATE -O0 -ggdb)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
endif()

# CUDA-specific extra flags
target_compile_options(Thor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings -Xcompiler -fPIC>
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
target_link_libraries(thor_tests PRIVATE Thor GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(thor_tests)

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
        COMMAND find . -name "*.h" | grep -v ./googletest/ | grep -v ./build/ | grep -v ./test/ > ${CMAKE_BINARY_DIR}/headerlist.txt
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
