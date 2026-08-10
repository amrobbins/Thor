# API Reference

Thor's public API reference is Python-only for the initial release.

The [Python API](generated/) is generated from the curated public `thor` package surface on every documentation build. No list of classes, functions, or public namespaces is copied by hand into the reference.

Generation uses Python source and Griffe static analysis without importing `thor`, loading `_thor.so`, initializing CUDA, or compiling Thor. Pure-Python definitions and re-exports are documented immediately. Native-backed namespaces automatically expand when nanobind-generated `.pyi` metadata is available under `docs/_api_stubs/thor-stubs`.

This means API-reference structure follows the shipped Python surface while narrative documentation remains intentionally hand-authored. C++ implementation APIs are not part of the initial public documentation surface.
