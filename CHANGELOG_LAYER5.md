# CHANGELOG — Layer 5 (Build)

## Fixed

### 1. Diagnostic regex `$$$$` instead of escaped parentheses
- **Bug:** `_DIAG_RE` used `$$` where `\(` and `` were intended. In Python regex,
  `$` is the end-of-line anchor, so the pattern could never match nvcc's
  `file(line): severity: message` format.
- **Fix:** Replaced `$$$$` with `$$` and `$$`.
- **File:** `nvcc_runner.py`

### 2. Executable vs cubin semantically overloaded in CompilationResult
- **Bug:** `compile_executable()` stored the runnable binary in `CompilationResult.cubin_path`.
  Downstream code (ArtifactManager, result_repo) could not distinguish an executable
  from a cubin, causing `store_compilation_result()` to label executables as `kind="cubin"`.
- **Fix:** Added `binary_path: str | None = None` to `CompilationResult` (backward compatible).
  `compile_executable()` now sets `binary_path` and nullifies `cubin_path`.
  `ArtifactManager.store_compilation_result()` checks `binary_path` and stores as `kind="executable"`.
- **Files:** `nvcc_runner.py`, `artifact_manager.py`, `schemas/compilation.py` (schema addition)

### 3. `cleanup_variant()` used manual path construction
- **Bug:** Cleanup called `base_dir / rec.rel_path` and `unlink()` directly, bypassing
  ArtifactStore's path-resolution logic. If the store's layout ever changes, cleanup breaks.
- **Fix:** Added `ArtifactStore.delete(rel_path)` method. `cleanup_variant()` now calls
  `self._store.delete(rec.rel_path)`.
- **Files:** `artifact_manager.py`, `artifact_store.py`

### 4. `common_flags()` accepted invalid optimization levels
- **Bug:** Any integer was accepted for `optimization_level`, producing flags like `-O99`
  that nvcc rejects silently or with obscure errors.
- **Fix:** Validation — raises `ConfigurationError` if level not in {0, 1, 2, 3}.
- **File:** `arch_flags.py`

## Added

### `ArtifactStore.delete(rel_path)`
- Single-file deletion method for consistent path resolution.
- No-op if file does not exist (safe for cleanup).
- **File:** `artifact_store.py`

### `CompilationResult.binary_path` field
- `str | None`, default `None`. Populated by `compile_executable()` only.
- Backward compatible — existing code using `cubin_path` is unaffected.
- **File:** `schemas/compilation.py`

### Mock-based integration tests (`test_integration.py`)
- 9 tests covering full pipeline without real nvcc:
  - `TestMockCompileToCubin` — success + failure paths
  - `TestMockExtractPTX` — success + PTXExtractionError
  - `TestMockCompileExecutable` — binary_path set, cubin_path null
  - `TestMockFullPipeline` — compile→store→retrieve→cleanup for cubin, PTX, executable
  - `TestFlagsIntegration` — flag combinations, invalid level rejection

### Additional unit tests
- `test_arch_flags.py`: +3 tests (executable flag, valid levels, invalid level rejection)
- `test_nvcc_runner.py`: +3 tests (severity code, info severity, executable binary_path)
- `test_artifact_manager.py`: +3 tests (binary_path storage, no-cubin-no-binary, nonexistent variant cleanup)

## Preserved

- Async subprocess architecture (asyncio.create_subprocess_exec)
- Repository pattern and schema contracts
- Diagnostic parsing structure (three-pass: standard, fatal, linker)
- Compilation timeout (120 seconds)
- ArtifactStore content-addressable deduplication
- All existing test behavior (no tests removed)


Summary of All File Changes

File	Changed?	What
build/__init__.py	No	Already correct from v2
build/arch_flags.py	Yes	Added optimization_level validation in common_flags()
build/nvcc_runner.py	Yes	Fixed _DIAG_RE regex ($$ -> $$$$); compile_executable() sets binary_path, nulls cubin_path
build/artifact_manager.py	Yes	cleanup_variant() uses store.delete(); store_compilation_result() handles binary_path
memory/artifact_store.py	Yes	Added delete(rel_path) method
schemas/compilation.py	Yes	Added binary_path: str | None = None (backward compatible)
tests/test_arch_flags.py	Yes	+3 tests
tests/test_nvcc_runner.py	Yes	+3 tests
tests/test_artifact_manager.py	Yes	+3 tests
tests/test_integration.py	Yes	New file: 9 mock-based integration tests
CHANGELOG_LAYER5.md	Yes	New file
