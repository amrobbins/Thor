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

After a normal Thor Python build has generated its stubs, synchronize them into the documentation metadata snapshot with:

```bash
python tools/docs/sync_api_stubs.py build/bindings/python/thor
```

Pass the actual generated `thor` package directory if your build directory has a different name. The source directory must contain both `__init__.pyi` and `_thor.pyi`. The sync tool copies only `.pyi` files and never imports Thor.

To verify a checked-in snapshot against a generated stub tree without modifying files, use:

```bash
python tools/docs/sync_api_stubs.py build/bindings/python/thor --check
```

The docs configuration also enables Griffe's stub-package lookup and includes `docs/_api_stubs` on its search path. Once a generated `thor-stubs` snapshot is present, native-backed public symbols can be rendered from that metadata while `allow_inspection = false` continues to prevent `_thor` from being imported.

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
