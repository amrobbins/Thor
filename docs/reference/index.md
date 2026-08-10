# API Reference

Thor's public API reference is Python-only for the initial release.

Reference pages are rendered with mkdocstrings and Griffe from Thor's public Python API metadata. Griffe is configured for static analysis only: documentation builds must never import `thor._thor`, initialize CUDA, or compile Thor.

The first reference pages intentionally prove that pipeline on public objects whose definitions live in Python source. Native-backed objects such as `thor.Tensor` and `thor.Network` will join the same reference surface once their generated `.pyi` metadata is made available to the docs build. Their signatures will not be copied by hand into Markdown.

- [Core API](core.md) currently documents the Python-defined `thor.einsum` convenience function.
- [Ensembles](ensembles.md) exercises class, method, property, and type-annotation rendering from Thor's Python package.

As the reference expands, the curated `thor` namespaces remain the public boundary. C++ implementation APIs are not part of the initial documentation surface.
