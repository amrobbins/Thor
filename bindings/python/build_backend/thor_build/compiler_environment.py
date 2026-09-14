"""Host compiler selection shared by Thor wheel build backends."""

from __future__ import annotations

import os
import shutil


def configure_host_compiler_environment() -> None:
    """Prefer Thor's GCC 14 host toolchain for isolated wheel builds.

    scikit-build-core may seed CMake's compiler cache before Thor's top-level
    CMake policy runs. Selecting the compilers in the PEP 517 environment
    keeps wheel builds consistent with Thor's normal GCC 14 auto-selection
    while still respecting explicit CC/CXX/CUDAHOSTCXX choices.
    """
    if not os.environ.get("CC"):
        gcc14 = shutil.which("gcc-14")
        if gcc14:
            os.environ["CC"] = gcc14

    if not os.environ.get("CXX"):
        gxx14 = shutil.which("g++-14")
        if gxx14:
            os.environ["CXX"] = gxx14

    if not os.environ.get("CUDAHOSTCXX") and os.environ.get("CXX"):
        os.environ["CUDAHOSTCXX"] = os.environ["CXX"]
