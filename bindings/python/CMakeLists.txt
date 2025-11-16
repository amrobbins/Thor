cmake_minimum_required(VERSION 3.22)

# 1) Find Python from the active venv, including the module-development bits
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# 2) Ask Python where pybind11's CMake config lives (inside the venv)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${pybind11_DIR}" CACHE PATH "Path to pybind11 CMake config")

# 3) Load pybind11 and its CMake functions
find_package(pybind11 CONFIG REQUIRED)
target_link_libraries(Thor PRIVATE pybind11::headers)

# 4) Build the extension module
file(GLOB_RECURSE THOR_PY_BIND_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        )

pybind11_add_module(thor
        ${THOR_PY_BIND_SOURCES}
        )

# 5) Link to the core Thor library
target_link_libraries(thor PRIVATE Thor)
target_compile_features(thor PRIVATE cxx_std_20)
