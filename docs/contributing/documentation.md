# Documentation development

Thor's documentation is a standalone static-site build. It must remain possible to edit and validate narrative documentation without compiling Thor or requiring a CUDA-capable machine.

## Set up the documentation environment

From the repository root, create a dedicated virtual environment and install the pinned documentation dependency:

```bash
python3 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r requirements/docs.txt
```

Keeping the documentation environment separate from Thor's development environment prevents documentation-only work from changing the CUDA/Python build dependency graph.

## Preview locally

```bash
zensical serve
```

The development server rebuilds the site as documentation files change and serves it on `http://localhost:8000` by default.

## Validate a production build

```bash
zensical build --clean --strict
```

The generated static site is written to `site/`. The `--strict` build is the canonical documentation check and should be used by CI. Internal links and anchors are validated during the build.

## Source-of-truth rules

- Narrative documentation belongs under `docs/`.
- Public API signatures must come from the released Python API metadata rather than copied signatures in Markdown.
- Runnable examples should have one canonical source file that is both tested and included in the documentation.
- C++ implementation APIs are not part of the initial public documentation surface.

## Continuous integration

The `Documentation` GitHub Actions workflow runs for every pull request and every push to `master`. It installs only the dependencies in `requirements/docs.txt` and runs the same strict production build used locally:

```bash
zensical build --clean --strict
```

The documentation CI job intentionally does not configure CUDA, build Thor, or import Thor's native extension. Narrative documentation must remain independently buildable. API-reference validation that depends on released Python metadata is added separately so documentation-only changes stay lightweight.

## API reference generation

API reference pages use mkdocstrings with the Python handler. Griffe reads Thor's package source from `bindings/python/src` using static analysis; `allow_inspection` is disabled globally for the documentation build.

That restriction is intentional. A documentation-only build must not import `thor`, load `_thor.so`, initialize the CUDA runtime, or require a Thor build tree. Python-defined public objects can therefore be documented directly from source, while native-backed objects must be documented from generated `.pyi` metadata once that metadata is supplied to the docs build.

Reference Markdown should contain mkdocstrings directives rather than copied signatures. For example:

```text
::: thor.einsum
```

Do not hand-maintain a native signature in Markdown as a substitute for generated stub metadata. The released Python API remains the source of truth.

### Native stub metadata

Normal source-tree CMake builds keep the documentation stub snapshot synchronized automatically. Once nanobind has generated the root and public namespace `.pyi` files, the `thor_docs_api_stub_snapshot` target copies that exact tree into `docs/_api_stubs/thor-stubs` and regenerates `docs/reference/generated/`. The default source-tree build depends on this target when `THOR_PYTHON_SYNC_DOC_STUBS=ON`, which is the default when the documentation tooling is present.

This does **not** make documentation CI a native build. The CUDA-free docs jobs only consume the checked-in snapshot. Isolated scikit-build wheel builds default `THOR_PYTHON_SYNC_DOC_STUBS` to `OFF` so packaging cannot modify the source checkout; those builds still generate and install the same `.pyi` files into the wheel.

To force a snapshot refresh from an existing build tree:

```bash
cmake --build build --target thor_docs_api_stub_snapshot
```

To verify that both the checked-in snapshot and generated API-reference Markdown exactly match the current native bindings without modifying either:

```bash
cmake --build build --target thor_docs_api_stub_snapshot_check
```

If a source-tree build must not update documentation metadata, configure it with `-DTHOR_PYTHON_SYNC_DOC_STUBS=OFF`. The lower-level `tools/docs/sync_api_stubs.py` command remains available for unusual build-directory layouts, but routine development should not need to invoke it directly.

The docs configuration enables Griffe's stub-package lookup and includes `docs/_api_stubs` on its search path. Once the generated `thor-stubs` snapshot is checked in, native-backed public symbols are rendered from that metadata while `allow_inspection = false` continues to prevent `_thor` from being imported.

## Generated Python API reference

The namespace reference under `docs/reference/generated/` is generated. Do not hand-edit those files. Regenerate it from the repository root with:

```bash
python tools/docs/generate_api_reference.py
```

The generator reads Thor's curated Python package surface without importing `thor`. It documents statically resolvable Python objects immediately and uses the checked-in nanobind `.pyi` snapshot when one is present. Native-backed pages therefore fill in automatically when stub metadata becomes available.

Both documentation GitHub Actions workflows run the generator before validating and building the site, so the published GitHub Pages reference always reflects the API metadata available in that commit. The generated files are also kept in the repository so `zensical serve` works immediately after checkout. To verify that the checked-in generated files are current, run:

```bash
python tools/docs/generate_api_reference.py --check
```

## Public API documentation inventory

Run the static API inventory check from the repository root with:

```bash
python tools/docs/check_public_api.py
```

The checker never imports `thor`. It reads literal public `__all__` declarations from the Python package, supplements native-only namespaces from the checked-in generated stub snapshot when one is available, and compares that inventory with mkdocstrings directives under `docs/reference/`.

During scaffolding, missing API pages are reported but do not fail CI because the reference is intentionally being filled in incrementally. CI does fail for reference directives that are known not to be public, malformed mkdocstrings directives, and `TODO`/`FIXME` markers in API-reference Markdown.

Once the native stub snapshot and reference surface are complete, the release gate can require full coverage with:

```bash
python tools/docs/check_public_api.py --require-complete
```

That mode additionally fails for every resolved public symbol without a reference directive and for any public namespace whose native metadata is unavailable.

## Publishing

Production publication is intentionally separate from pull-request validation so the ordinary `Documentation` workflow can keep read-only repository permissions.

The `Publish documentation` workflow rebuilds the same strict site on pushes to `master`, uploads the generated `site/` directory as a GitHub Pages artifact, and deploys it to the `github-pages` environment. It can also be run manually with `workflow_dispatch`.

Before the first deployment, configure the repository once in **Settings → Pages → Build and deployment → Source** and select **GitHub Actions**. No generated HTML is committed to the repository.
