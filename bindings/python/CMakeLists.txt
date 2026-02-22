cmake_minimum_required(VERSION 3.22)

# ---- Options ----
option(THOR_PYTHON_TESTS "Enable Python binding tests (pytest)" ON)
option(THOR_PYTHON_STUBS "Generate typing stubs via nanobind_add_stub()" ON)

# If ON (and not building under scikit-build-core), `cmake --build .` will also copy into the venv.
option(THOR_PYTHON_AUTO_INSTALL "Auto-install bindings into the project venv as part of the default build" ON)

# ---- Python interpreter (parent/root should have set Python3_EXECUTABLE to your .venv python) ----
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Determine site-packages paths for the chosen interpreter
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
        OUTPUT_VARIABLE THOR_PY_PLATLIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
        OUTPUT_VARIABLE THOR_PY_PURELIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Thor: venv platlib = ${THOR_PY_PLATLIB}")
message(STATUS "Thor: venv purelib = ${THOR_PY_PURELIB}")

set(THOR_PY_PKG_NAME "thor")
set(THOR_PY_PKG_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/${THOR_PY_PKG_NAME}")
set(THOR_PY_PKG_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/${THOR_PY_PKG_NAME}")
set(THOR_PY_VENV_DIR "${THOR_PY_PLATLIB}/${THOR_PY_PKG_NAME}")

# ---- Sync Python package into the build tree (so the built extension lives inside a real package dir) ----
add_custom_command(
        OUTPUT "${THOR_PY_PKG_BUILD_DIR}/.thor_pkg_sync.stamp"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${THOR_PY_PKG_BUILD_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${THOR_PY_PKG_SRC_DIR}" "${THOR_PY_PKG_BUILD_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${THOR_PY_PKG_BUILD_DIR}/.thor_pkg_sync.stamp"
        DEPENDS
        "${THOR_PY_PKG_SRC_DIR}/__init__.py"
        "${THOR_PY_PKG_SRC_DIR}/py.typed"
        COMMENT "Thor: syncing python package sources into build tree"
)
add_custom_target(thor_py_pkg_sync DEPENDS "${THOR_PY_PKG_BUILD_DIR}/.thor_pkg_sync.stamp")

# ---- Collect all binding sources (C++ only) ----
file(GLOB_RECURSE THOR_PY_BIND_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# ---- Build native extension: thor/_thor*.so ----
# NB_MODULE name in C++ must match this target name. :contentReference[oaicite:6]{index=6}
nanobind_add_module(_thor
        NB_STATIC
        ${THOR_PY_BIND_SOURCES}
)
#target_link_options(_thor PRIVATE "LINKER:-rpath,\$ORIGIN")
install(TARGETS Thor
        LIBRARY DESTINATION thor
)

add_dependencies(_thor thor_py_pkg_sync)

target_link_libraries(_thor PRIVATE Thor)
target_compile_features(_thor PRIVATE cxx_std_20)

# Put the built extension inside the build-tree package dir: <bin>/bindings/python/thor/_thor*.so
set_target_properties(_thor PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${THOR_PY_PKG_BUILD_DIR}"
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

set(THOR_PYTHON_PACKAGE_DIR "thor")

# Install the zlib-ng runtime shared library next to the extension/libThor.so
get_target_property(_zng_type ${ZLIBNG_TARGET} TYPE)
if (_zng_type STREQUAL "SHARED_LIBRARY")
    install(TARGETS ${ZLIBNG_TARGET}
            LIBRARY DESTINATION ${THOR_PYTHON_PACKAGE_DIR}
    )
endif ()

target_include_directories(_thor PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# ---- venv install target: copy package + extension into .venv site-packages ----
if (THOR_PYTHON_AUTO_INSTALL AND NOT DEFINED SKBUILD_PROJECT_NAME)
    add_custom_target(thor_py_install ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory "${THOR_PY_VENV_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${THOR_PY_PKG_SRC_DIR}" "${THOR_PY_VENV_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:_thor>" "${THOR_PY_VENV_DIR}/"
            DEPENDS _thor
            COMMENT "Thor: installing Python package + native extension into venv"
    )
else ()
    add_custom_target(thor_py_install
            COMMAND ${CMAKE_COMMAND} -E make_directory "${THOR_PY_VENV_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${THOR_PY_PKG_SRC_DIR}" "${THOR_PY_VENV_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:_thor>" "${THOR_PY_VENV_DIR}/"
            DEPENDS _thor
            COMMENT "Thor: installing Python package + native extension into venv"
    )
endif ()

# ---- Python deps from pyproject.toml, with a stamp file ----
set(THOR_PYPROJECT_TOML "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")
set(THOR_PY_DEPS_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install_py_deps.py")
set(THOR_PY_DEPS_STAMP "${CMAKE_CURRENT_BINARY_DIR}/thor_py_deps.stamp")

# Generate a small helper script at configure time (no quoting issues at build time)
file(WRITE "${THOR_PY_DEPS_SCRIPT}" [=[
import sys
import pathlib
import subprocess

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # fallback if ever needed

def main() -> int:
    pyproject = pathlib.Path(sys.argv[1]).resolve()
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    deps = list(data.get("project", {}).get("dependencies", []))
    deps += list(data.get("project", {}).get("optional-dependencies", {}).get("dev", []))

    # Ensure pip tooling is current
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    # Install deps (no build/install of the package itself)
    if deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
]=])

add_custom_command(
        OUTPUT "${THOR_PY_DEPS_STAMP}"
        COMMAND "${Python3_EXECUTABLE}" "${THOR_PY_DEPS_SCRIPT}" "${THOR_PYPROJECT_TOML}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${THOR_PY_DEPS_STAMP}"
        DEPENDS "${THOR_PYPROJECT_TOML}"
        COMMENT "Thor: installing Python deps from pyproject.toml (dependencies + dev) into venv"
        VERBATIM
)

add_custom_target(thor_py_deps
        DEPENDS "${THOR_PY_DEPS_STAMP}"
)

# ---- Stub generation ----
# Generates build-tree: <bin>/bindings/python/thor/_thor.pyi and py.typed marker if requested. :contentReference[oaicite:7]{index=7}
if (THOR_PYTHON_STUBS)
    nanobind_add_stub(
            thor_py_stub
            MODULE thor._thor
            OUTPUT "${THOR_PY_PKG_BUILD_DIR}/_thor.pyi"
            MARKER_FILE "${THOR_PY_PKG_BUILD_DIR}/py.typed"
            PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS _thor
    )
    add_dependencies(thor_py_stub thor_py_pkg_sync)

    add_custom_target(thor_py_stub_install
            COMMAND ${CMAKE_COMMAND} -E copy "${THOR_PY_PKG_BUILD_DIR}/_thor.pyi" "${THOR_PY_VENV_DIR}/_thor.pyi"
            COMMAND ${CMAKE_COMMAND} -E copy "${THOR_PY_PKG_BUILD_DIR}/py.typed" "${THOR_PY_VENV_DIR}/py.typed"
            COMMAND ${CMAKE_COMMAND} -E copy "${THOR_PY_PKG_BUILD_DIR}/__init__.pyi" "${THOR_PY_VENV_DIR}/__init__.pyi"
            DEPENDS thor_py_stub
            COMMENT "Thor: installing generated stubs into venv"
    )

    # Generate stubs for the public wrapper modules so IDEs see thor.layers.*, etc.
    set(THOR_PY_PUBLIC_SUBMODULES activations initializers layers losses metrics optimizers)

    set(_thor_public_stub_targets "")
    foreach (_sm IN LISTS THOR_PY_PUBLIC_SUBMODULES)
        nanobind_add_stub(
                thor_py_stub_${_sm}
                MODULE thor.${_sm}
                OUTPUT "${THOR_PY_PKG_BUILD_DIR}/${_sm}/__init__.pyi"
                PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}"
                DEPENDS _thor
        )
        add_dependencies(thor_py_stub_${_sm} thor_py_pkg_sync)
        list(APPEND _thor_public_stub_targets thor_py_stub_${_sm})
    endforeach ()

    # Install those wrapper stubs into the venv
    add_custom_target(thor_py_public_stub_install
            COMMAND ${CMAKE_COMMAND} -E echo "Thor: installing public submodule stubs into venv"
            DEPENDS ${_thor_public_stub_targets}
    )

    foreach (_sm IN LISTS THOR_PY_PUBLIC_SUBMODULES)
        add_custom_command(TARGET thor_py_public_stub_install POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${THOR_PY_VENV_DIR}/${_sm}"
                COMMAND ${CMAKE_COMMAND} -E copy
                "${THOR_PY_PKG_BUILD_DIR}/${_sm}/__init__.pyi"
                "${THOR_PY_VENV_DIR}/${_sm}/__init__.pyi"
        )
    endforeach ()

    nanobind_add_stub(
            thor_py_stub_thor
            MODULE thor
            OUTPUT "${THOR_PY_PKG_BUILD_DIR}/__init__.pyi"
            PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS _thor
    )
    add_dependencies(thor_py_stub_thor thor_py_pkg_sync)

    # And make the main stub install depend on it
    add_dependencies(thor_py_stub_install thor_py_stub_thor)
    add_dependencies(thor_py_stub_install thor_py_public_stub_install)
    add_dependencies(thor_py_install thor_py_stub_install)

    # Aggregate target so we can easily depend on "all stubs"
    add_custom_target(thor_py_stubs_all
            DEPENDS
            thor_py_stub
            thor_py_stub_thor
            ${_thor_public_stub_targets}
    )

    # IMPORTANT: for wheel builds, ensure stubs are generated as part of the default build
    if (DEFINED SKBUILD_PROJECT_NAME)
        add_dependencies(_thor thor_py_pkg_sync) # you already have this
        add_custom_target(thor_py_stubs_all_build ALL DEPENDS thor_py_stubs_all)
    endif ()
endif ()

# ---- Pytest target + optional CTest ----
if (THOR_PYTHON_TESTS)
    set(THOR_PY_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")

    add_custom_target(thor_pytest
            COMMAND "${Python3_EXECUTABLE}" -m pytest -q "${THOR_PY_TEST_DIR}"
            DEPENDS thor_py_install thor_py_deps
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Thor: running Python binding tests"
    )

    include(CTest)
    add_test(
            NAME thor_python_pytest
            COMMAND "${Python3_EXECUTABLE}" -m pytest -q "${THOR_PY_TEST_DIR}"
    )
endif ()

# ---- Install rules for wheel builds (scikit-build-core) ----
# scikit-build-core sets SKBUILD_PROJECT_NAME; DESTINATION should be the package dir. :contentReference[oaicite:8]{index=8}
if (DEFINED SKBUILD_PROJECT_NAME)
    set(_thor_install_pkg_dir "${SKBUILD_PROJECT_NAME}")
else ()
    set(_thor_install_pkg_dir "${THOR_PY_PKG_NAME}")
endif ()

# Install the extension into the package directory (thor/_thor*.so)
install(TARGETS _thor
        LIBRARY DESTINATION "${_thor_install_pkg_dir}"
)

# Install the pure python package sources into the same package dir
install(DIRECTORY "${THOR_PY_PKG_SRC_DIR}/"
        DESTINATION "${_thor_install_pkg_dir}"
)

# If stubs were generated in the build tree, install them too
if (THOR_PYTHON_STUBS)
    install(FILES
            "${THOR_PY_PKG_BUILD_DIR}/_thor.pyi"
            "${THOR_PY_PKG_BUILD_DIR}/__init__.pyi"
            "${THOR_PY_PKG_BUILD_DIR}/py.typed"
            DESTINATION "${_thor_install_pkg_dir}"
    )

    # Submodule stubs
    foreach (_sm IN LISTS THOR_PY_PUBLIC_SUBMODULES)
        install(FILES
                "${THOR_PY_PKG_BUILD_DIR}/${_sm}/__init__.pyi"
                DESTINATION "${_thor_install_pkg_dir}/${_sm}"
        )
    endforeach ()
endif ()
