# CHANGELOG — Layer 1 (Memory / Database)

## Fixed

### Repository boundary violation: RunRepo → KernelRepo private coupling
- **Before:** `RunRepo.save_run()` called `self._kernel_repo._resolve_kernel_id()` (a private method prefixed with `_`).
- **After:** Renamed to `resolve_kernel_id()` (public method) in `KernelRepo`. `RunRepo` calls the public API.
- **Files:** `memory/repositories/kernel_repo.py`, `memory/repositories/run_repo.py`

### Hidden repository instantiation in RunRepo
- **Before:** `RunRepo.__init__` internally created its own `KernelRepo(session)`, ignoring any KernelRepo the caller may already hold. Both repos shared the same session but had independent lifecycles.
- **After:** `RunRepo.__init__` accepts an optional `kernel_repo: KernelRepo` parameter. When provided, it uses the caller's instance. Falls back to creating one internally for backward compatibility.
- **Files:** `memory/repositories/run_repo.py`

### Invalid schema stub on missing variant
- **Before:** `RunRepo._iteration_to_pydantic` constructed a `KernelSource(name="unknown", source_code="")` stub when a variant could not be loaded. This leaked ORM internals into schema consumers and produced objects that violated schema contracts.
- **After:** Raises `StorageError` with a descriptive message when the referenced variant cannot be loaded.
- **Files:** `memory/repositories/run_repo.py`

### Missing None guard in MetricsRepo.find_by_variant
- **Before:** `find_by_variant` passed the row directly to `_to_pydantic` without checking for `None`.
- **After:** Explicitly returns `None` when no row is found.
- **Files:** `memory/repositories/metrics_repo.py`

### Missing None guard in ResultRepo.find_correctness_by_variant
- **Before:** `find_correctness_by_variant` passed the row to `_correctness_to_pydantic` without checking for `None`.
- **After:** Explicitly returns `None` when no row is found.
- **Files:** `memory/repositories/result_repo.py`

## Added

### KernelRepo.update_source()
- New public method to update mutable fields on an existing `KernelSource` row by DB id.
- **File:** `memory/repositories/kernel_repo.py`

### Test suite (`tests/unit/test_memory.py`)
- **38 tests** covering all required scenarios:
  - `TestCreateKernel` — save, get, find, list sources and variants; variant with transform history
  - `TestUpdateKernel` — update source name/code, update run status
  - `TestCreateRun` — save/get run, auto-persist kernel, list with filters
  - `TestCreateIteration` — save/list iterations, FK detail references, unloadable variant error
  - `TestTransactionRollback` — rollback discards data, rollback on exception, committed data persists
  - `TestArtifactPersistence` — text/binary store, dedup, exists, size
  - `TestArtifactRetrieval` — not found, UTF-8 decode, binary round-trip
  - `TestDeleteBehavior` — cleanup single/all runs, cascade delete kernel→variants, cascade delete run→iterations, SET NULL on variant delete
  - `TestConcurrentSessionBehavior` — cross-session visibility, uncommitted isolation, threaded writes
  - `TestConversionCorrectness` — round-trip for all result types (metrics, compilation, correctness, benchmark, transform)
  - `TestRepositoryBoundaries` — injected kernel repo, public API usage
  - `TestEndToEnd` — full pipeline integration test

### Test fixtures
- Added `db_session_factory` fixture for tests requiring multiple independent sessions.
- **File:** `tests/conftest.py`

## Preserved

- Repository pattern (no architecture changes)
- Schema contracts (Pydantic schemas unchanged)
- ORM model definitions (no migration work)
- `BaseRepository` generic CRUD pattern
- `ArtifactStore` content-addressable deduplication
- Session lifecycle managed by callers via `get_session()`
