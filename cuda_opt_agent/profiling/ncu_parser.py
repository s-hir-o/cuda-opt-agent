
"""Parse NCU raw metric output into ProfileMetrics.

Supports two input formats:
1. Dict-based: {metric_name: value} (from JSON import)
2. CSV text: ncu --page raw --csv output
"""

from __future__ import annotations

import csv
import io
from typing import Any

from cuda_opt_agent.core.exceptions import MetricsParseError
from cuda_opt_agent.schemas.metrics import ProfileMetrics

# NCU metric name → ProfileMetrics field mapping
_METRIC_MAP: dict[str, str] = {
    "gpu__time_duration.sum": "gpu_time_ns",
    "sm__throughput.avg_pct_of_peak_sustained_elapsed": "sm_throughput",
    "dram__throughput.avg_pct_of_peak_sustained_elapsed": "dram_throughput",
    "l1tex__throughput.avg_pct_of_peak_sustained_elapsed": "l1_throughput",
    "lts__throughput.avg_pct_of_peak_sustained_elapsed": "l2_throughput",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_dynamic": "_shared_dynamic",
    "launch__shared_mem_per_block_static": "_shared_static",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "global_load_efficiency",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct": "global_store_efficiency",
    "lts__t_sectors_hitrate.pct": "l2_hit_rate",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
}

# Metadata metrics — extracted but not mapped to ProfileMetrics scalar fields
_METADATA_METRICS: dict[str, str] = {
    "gpu_name": "gpu_name",
    "device__attribute_architecture_name": "arch",
}

# Column name pairs across NCU output formats
_COLUMN_ALIASES: list[tuple[str, str]] = [
    ("Metric Name", "metric_name"),
    ("Metric Value", "metric_value"),
    ("Metric Unit", "metric_unit"),
]


def parse_ncu_csv(csv_text: str, kernel_name: str = "") -> ProfileMetrics:
    """Parse NCU CSV output into ProfileMetrics.

    Expects the ncu --page raw --csv format with columns:
    "Metric Name", "Metric Value", "Metric Unit" (and optionally others).

    Handles alternative column names via case-insensitive alias matching.

    Args:
        csv_text: Raw NCU CSV output.
        kernel_name: Optional kernel name to embed in metrics.

    Returns:
        Populated ProfileMetrics.

    Raises:
        MetricsParseError: If the CSV is empty or unparseable.
    """
    raw_metrics: dict[str, Any] = {}
    gpu_name = ""
    arch = ""

    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise MetricsParseError("Empty or unparseable NCU CSV output")

    # Normalize field names: build alias lookup
    field_aliases = _build_field_aliases(reader.fieldnames)

    name_col = field_aliases.get("name")
    value_col = field_aliases.get("value")
    if name_col is None or value_col is None:
        raise MetricsParseError(
            f"NCU CSV missing required columns. Found: {list(reader.fieldnames)}"
        )

    for row in reader:
        name = (row.get(name_col) or "").strip()
        value = (row.get(value_col) or "").strip()
        if not name or not value:
            continue
        try:
            numeric = float(value.replace(",", ""))
        except ValueError:
            numeric = value
        raw_metrics[name] = numeric

    if not raw_metrics:
        raise MetricsParseError("NCU CSV contains no parseable metric rows")

    # Extract GPU metadata if present
    for meta_key, target_field in _METADATA_METRICS.items():
        val = raw_metrics.get(meta_key)
        if val is not None and isinstance(val, str):
            if target_field == "gpu_name":
                gpu_name = val
            elif target_field == "arch":
                arch = val

    return _build_metrics(raw_metrics, kernel_name=kernel_name, gpu_name=gpu_name, arch=arch)


def parse_ncu_dict(
    metrics: dict[str, Any],
    kernel_name: str = "",
    gpu_name: str = "",
    arch: str = "",
) -> ProfileMetrics:
    """Parse a flat dict of NCU metric names → values."""
    # Extract metadata from dict keys if not explicitly provided
    if not gpu_name:
        val = metrics.get("gpu_name")
        if val is not None:
            gpu_name = str(val)
    if not arch:
        val = metrics.get("device__attribute_architecture_name")
        if val is not None:
            arch = str(val)

    return _build_metrics(metrics, kernel_name=kernel_name, gpu_name=gpu_name, arch=arch)


def _build_field_aliases(fieldnames: list[str]) -> dict[str, str]:
    """Build a lookup from canonical role ('name', 'value') to actual column name."""
    aliases: dict[str, str] = {}
    normalized = {f.lower().strip(): f for f in fieldnames}

    for canonical, alt in _COLUMN_ALIASES:
        for candidate in (canonical, alt):
            lower = candidate.lower()
            if lower in normalized:
                if canonical == "Metric Name":
                    aliases.setdefault("name", normalized[lower])
                elif canonical == "Metric Value":
                    aliases.setdefault("value", normalized[lower])
                elif canonical == "Metric Unit":
                    aliases.setdefault("unit", normalized[lower])

    return aliases


def _build_metrics(
    raw: dict[str, Any],
    kernel_name: str,
    gpu_name: str,
    arch: str,
) -> ProfileMetrics:
    """Map raw NCU metrics to ProfileMetrics fields."""
    mapped: dict[str, Any] = {}
    shared_dynamic = 0
    shared_static = 0

    for ncu_name, field in _METRIC_MAP.items():
        val = raw.get(ncu_name)
        if val is None:
            continue
        # Skip non-numeric string values (e.g., "N/A")
        if isinstance(val, str):
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
        if field == "_shared_dynamic":
            shared_dynamic = int(val) if isinstance(val, (int, float)) else 0
        elif field == "_shared_static":
            shared_static = int(val) if isinstance(val, (int, float)) else 0
        else:
            mapped[field] = val

    

    # Convert ns → µs if we have ns
    if "gpu_time_ns" in mapped and isinstance(mapped["gpu_time_ns"], (int, float)):
        mapped["gpu_time_us"] = mapped["gpu_time_ns"] / 1000.0

    # Occupancy → fraction if > 1 (NCU sometimes reports as percentage)
    occ = mapped.get("achieved_occupancy")
    if isinstance(occ, (int, float)) and occ > 1.0:
        mapped["achieved_occupancy"] = occ / 100.0

    # Occupancy limiters
    for limiter_ncu, field in [
        ("launch__occupancy_limit_registers", "occupancy_limited_by_registers"),
        ("launch__occupancy_limit_shared_mem", "occupancy_limited_by_shared_mem"),
        ("launch__occupancy_limit_block_size", "occupancy_limited_by_block_size"),
    ]:
        val = raw.get(limiter_ncu)
        if val is not None:
            mapped[field] = float(val) > 0 if isinstance(val, (int, float)) else bool(val)

    # Parse block/grid sizes from tuple strings or individual dimension fields
    _parse_launch_dims(raw, "block", mapped)
    _parse_launch_dims(raw, "grid", mapped)

    shared_mem = shared_dynamic + shared_static
    if shared_mem > 0:
        mapped["shared_mem_bytes"] = shared_mem

    return ProfileMetrics(
        kernel_name=kernel_name,
        gpu_name=gpu_name,
        arch=arch,
        raw_metrics=raw,
        **{k: v for k, v in mapped.items() if k in ProfileMetrics.model_fields},
    )


def _parse_launch_dims(raw: dict[str, Any], prefix: str, mapped: dict[str, Any]) -> None:
    """Parse block or grid dimensions from NCU raw metrics.

    Handles:
    - Single tuple string: "launch__block_size" → "(16, 16, 1)"
    - Individual dimensions: "launch__block_size_x" → 16, etc.
    - List/tuple values
    """
    target = f"{prefix}_size"
    tuple_key = f"launch__{prefix}_size"

    # Try single tuple string first
    val = raw.get(tuple_key)
    if isinstance(val, str) and val.startswith("("):
        try:
            dims = tuple(int(x.strip()) for x in val.strip("()").split(","))
            mapped[target] = dims
            return
        except (ValueError, TypeError):
            pass
    elif isinstance(val, (list, tuple)):
        mapped[target] = tuple(val)
        return

    # Try individual dimension fields
    x = _safe_int(raw.get(f"launch__{prefix}_size_x"))
    y = _safe_int(raw.get(f"launch__{prefix}_size_y"))
    z = _safe_int(raw.get(f"launch__{prefix}_size_z"))
    if x is not None:
        mapped[target] = (x, y or 1, z or 1)


def _safe_int(val: Any) -> int | None:
    """Convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
