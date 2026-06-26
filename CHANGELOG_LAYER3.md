markdown
# CHANGELOG: Layer 3 — Profiling

## Bug Fixes

### `profiling/ncu_parser.py`

1. **Column alias resolution**: The original parser only handled two exact column name
   pairs (`"Metric Name"/"Metric Value"` and `"metric_name"/"metric_value"`). Added
   `_build_field_aliases()` with case-insensitive matching across all known NCU output
   format variants.

2. **Individual dimension fields not parsed**: The original `_build_metrics` only handled
   single-tuple format (`"launch__block_size" -> "(16, 16, 1)"`). Added `_parse_launch_dims()`
   which additionally handles:
   - Individual x/y/z fields: `launch__grid_size_x`, `launch__block_size_y`, etc.
   - List/tuple values directly in the metrics dict.
   - Graceful fallback: tuple string tried first, then individual dims.

3. **Empty data rows silently accepted**: CSV with headers but zero data rows now raises
   `MetricsParseError` instead of returning an empty `ProfileMetrics`.

4. **GPU name/arch not extracted**: Added `_METADATA_METRICS` mapping to extract
   `gpu_name` and `device__attribute_architecture_name` from raw NCU data into the
   corresponding `ProfileMetrics` fields. Both `parse_ncu_csv()` and `parse_ncu_dict()`
   now populate these fields when present.

## Enhancements

### `profiling/mock_profiler.py`

Added two new presets covering the four-quadrant bottleneck space:

| Preset | SM% | DRAM% | Occupancy | Registers | Limited by |
|---|---|---|---|---|---|
| `compute_bound_kernel` | 92 | 15 | 85% | 64 | — |
| `memory_bound_kernel` | 20 | 90 | 40% | 48 | registers |

These are used by cost model and bottleneck classifier tests.

### `profiling/metrics_validator.py`

Extended range checks to cover additional fields:
- `l1_throughput` and `l2_throughput` (must be in [0, 100])
- `global_load_efficiency` and `global_store_efficiency` (must be in [0, 100])
- `l1_hit_rate` and `l2_hit_rate` (must be in [0, 100])

### `profiling/__init__.py`

Added `parse_ncu_csv`, `parse_ncu_dict`, and `validate_metrics` to `__all__` exports.

## Tests Added

| File | Count | Coverage |
|---|---|---|
| `test_ncu_parser.py` | 28 | Valid CSV, partial CSV, minimal CSV, empty CSV, no data rows, missing columns, malformed values, comma-separated numbers, occupancy normalization (fraction + percentage), block/grid tuple parsing, individual dimension parsing, shared memory (dynamic/static/combined), occupancy limiters (true/false), GPU name extraction, raw metrics preservation, alternate column names, dict parsing (valid/empty/percentage/list dims/individual dims/unknown keys/string values/metadata) |
| `test_profiler.py` | 20 | MockProfiler availability, all 4 presets, unknown kernel fallback, custom arch/gpu_name, variant lookup, block/grid size, metadata, all presets pass validator, MetricsValidator (valid metrics, negative/zero GPU time, throughput over/negative, l1/l2 throughput range, occupancy over/negative, register count high/zero, load/store efficiency range, l1/l2 hit rate range, all-None metrics), serialization roundtrip (full/minimal), mock output JSON-serializable, tuple→list→tuple roundtrip |


File tree after changes:


text



text
cuda_opt_agent/profiling/
├── __init__.py                    ✅ Updated exports
├── profiler.py                    ✅ Unchanged (ABC)
├── ncu_runner.py                  ✅ Unchanged
├── ncu_parser.py                  ✅ Fixed: column aliases, dim parsing, GPU metadata extraction
├── mock_profiler.py               ✅ Added: 2 new presets
└── metrics_validator.py           ✅ Extended: l1/l2 throughput + efficiency + hit rate checks

tests/unit/test_profiling/
├── __init__.py                    ✅ New
├── test_ncu_parser.py             ✅ New — 28 tests
└── test_profiler.py               ✅ New — 20 tests

CHANGELOG_LAYER3.md                ✅ New
