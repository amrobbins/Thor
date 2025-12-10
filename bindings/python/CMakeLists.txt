cmake_minimum_required(VERSION 3.22)

# Collect all your binding sources
file(GLOB_RECURSE THOR_PY_BIND_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Build the Python extension module with nanobind
nanobind_add_module(thor
        NB_SHARED
        ${THOR_PY_BIND_SOURCES}
)

target_link_libraries(thor PRIVATE Thor)
target_compile_features(thor PRIVATE cxx_std_20)

set_target_properties(thor PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
