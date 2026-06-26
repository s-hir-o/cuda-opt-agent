
"""Tests for NCU metric parsing — CSV and dict formats."""

from __future__ import annotations

import pytest

from cuda_opt_agent.core.exceptions import MetricsParseError
from cuda_opt_agent.profiling.ncu_parser import parse_ncu_csv, parse_ncu_dict


# ── Fixtures ───────────────────────────────────────────────────

VALID_CSV = """\
"Metric Name","Metric Value","Metric Unit"
"gpu__time_duration.sum","123456","nsecond"
"sm__throughput.avg_pct_of_peak_sustained_elapsed","65.3","%"
"dram__throughput.avg_pct_of_peak_sustained_elapsed","42.1","%"
"sm__warps_active.avg.pct_of_peak_sustained_active","75.0","%"
"launch__registers_per_thread","32",""
"launch__shared_mem_per_block_dynamic","4096","byte"
"launch__shared_mem_per_block_static","512","byte"
"smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct","90.0","%"
"smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct","95.0","%"
"lts__t_sectors_hitrate.pct","82.0","%"
"l1tex__t_sector_hit_rate.pct","88.0","%"
"launch__block_size","(16, 16, 1)",""
"launch__grid_size","(64, 64, 1)",""
"launch__occupancy_limit_registers","0",""
"launch__occupancy_limit_shared_mem","0",""
"launch__occupancy_limit_block_size","0",""
"""

PARTIAL_CSV = """\
"Metric Name","Metric Value","Metric Unit"
"gpu__time_duration.sum","500000","nsecond"
"sm__throughput.avg_pct_of_peak_sustained_elapsed","80.0","%"
"""

MINIMAL_CSV = """\
"Metric Name","Metric Value","Metric Unit"
"gpu__time_duration.sum","10000","nsecond"
"""


# ── CSV Parser Tests ──────────────────────────────────────────

class TestParseNCUCSV:

    def test_valid_full_output(self):
        m = parse_ncu_csv(VALID_CSV, kernel_name="test_kernel")
        assert m.kernel_name == "test_kernel"
        assert m.gpu_time_ns == 123456.0
        assert m.gpu_time_us == pytest.approx(123.456)
        assert m.sm_throughput == pytest.approx(65.3)
        assert m.dram_throughput == pytest.approx(42.1)
        assert m.achieved_occupancy == pytest.approx(0.75)
        assert m.registers_per_thread == 32
        assert m.global_load_efficiency == pytest.approx(90.0)
        assert m.global_store_efficiency == pytest.approx(95.0)
        assert m.l2_hit_rate == pytest.approx(82.0)
        assert m.l1_hit_rate == pytest.approx(88.0)
        assert m.shared_mem_bytes == 4608  # 4096 + 512

    def test_unit_conversion_ns_to_us(self):
        m = parse_ncu_csv(VALID_CSV)
        assert m.gpu_time_us == pytest.approx(m.gpu_time_ns / 1000.0)

    def test_occupancy_percentage_normalized(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"sm__warps_active.avg.pct_of_peak_sustained_active","50.0","%"\n'
        )
        m = parse_ncu_csv(csv)
        assert m.achieved_occupancy == pytest.approx(0.50)

    def test_occupancy_fraction_preserved(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"sm__warps_active.avg.pct_of_peak_sustained_active","0.75",""\n'
        )
        m = parse_ncu_csv(csv)
        assert m.achieved_occupancy == pytest.approx(0.75)

    def test_block_size_tuple_parsed(self):
        m = parse_ncu_csv(VALID_CSV)
        assert m.block_size == (16, 16, 1)

    def test_grid_size_tuple_parsed(self):
        m = parse_ncu_csv(VALID_CSV)
        assert m.grid_size == (64, 64, 1)

    def test_partial_output(self):
        m = parse_ncu_csv(PARTIAL_CSV, kernel_name="partial")
        assert m.kernel_name == "partial"
        assert m.gpu_time_us == pytest.approx(500.0)
        assert m.sm_throughput == pytest.approx(80.0)
        assert m.dram_throughput is None
        assert m.achieved_occupancy is None
        assert m.registers_per_thread is None
        assert m.shared_mem_bytes is None

    def test_minimal_output(self):
        m = parse_ncu_csv(MINIMAL_CSV, kernel_name="minimal")
        assert m.kernel_name == "minimal"
        assert m.gpu_time_ns == 10000.0
        assert m.sm_throughput is None

    def test_empty_csv_raises(self):
        with pytest.raises(MetricsParseError, match="Empty or unparseable"):
            parse_ncu_csv("")

    def test_no_data_rows_raises(self):
        csv = '"Metric Name","Metric Value","Metric Unit"\n'
        with pytest.raises(MetricsParseError, match="no parseable metric rows"):
            parse_ncu_csv(csv)

    def test_missing_columns_raises(self):
        csv = '"Foo","Bar"\n"a","b"\n'
        with pytest.raises(MetricsParseError, match="missing required columns"):
            parse_ncu_csv(csv)

    def test_malformed_values_skipped(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"gpu__time_duration.sum","N/A","nsecond"\n'
            '"sm__throughput.avg_pct_of_peak_sustained_elapsed","60.0","%"\n'
        )
        m = parse_ncu_csv(csv)
        assert m.gpu_time_ns is None
        assert m.sm_throughput == pytest.approx(60.0)

    def test_values_with_commas(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"gpu__time_duration.sum","1,234,567","nsecond"\n'
        )
        m = parse_ncu_csv(csv)
        assert m.gpu_time_ns == pytest.approx(1234567.0)

    def test_raw_metrics_preserved(self):
        m = parse_ncu_csv(VALID_CSV)
        assert m.raw_metrics is not None
        assert len(m.raw_metrics) > 0
        assert "gpu__time_duration.sum" in m.raw_metrics

    def test_gpu_name_extraction(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"gpu_name","NVIDIA A100",""\n'
            '"gpu__time_duration.sum","1000","nsecond"\n'
        )
        m = parse_ncu_csv(csv, kernel_name="test")
        assert m.gpu_name == "NVIDIA A100"

    def test_occupancy_limiter_registers_true(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"launch__occupancy_limit_registers","2",""\n'
        )
        m = parse_ncu_csv(csv)
        assert m.occupancy_limited_by_registers is True

    def test_occupancy_limiter_zero(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"launch__occupancy_limit_registers","0",""\n'
        )
        m = parse_ncu_csv(csv)
        assert m.occupancy_limited_by_registers is False

    def test_individual_grid_dims(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"launch__grid_size_x","128",""\n'
            '"launch__grid_size_y","256",""\n'
            '"launch__grid_size_z","1",""\n'
        )
        m = parse_ncu_csv(csv)
        assert m.grid_size == (128, 256, 1)

    def test_individual_block_dims(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"launch__block_size_x","32",""\n'
            '"launch__block_size_y","8",""\n'
            '"launch__block_size_z","1",""\n'
        )
        m = parse_ncu_csv(csv)
        assert m.block_size == (32, 8, 1)

    def test_shared_mem_dynamic_only(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"launch__shared_mem_per_block_dynamic","2048","byte"\n'
        )
        m = parse_ncu_csv(csv)
        assert m.shared_mem_bytes == 2048

    def test_shared_mem_static_only(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"launch__shared_mem_per_block_static","1024","byte"\n'
        )
        m = parse_ncu_csv(csv)
        assert m.shared_mem_bytes == 1024

    def test_throughput_at_boundary(self):
        csv = (
            '"Metric Name","Metric Value","Metric Unit"\n'
            '"sm__throughput.avg_pct_of_peak_sustained_elapsed","100.0","%"\n'
        )
        m = parse_ncu_csv(csv)
        assert m.sm_throughput == pytest.approx(100.0)

    def test_kernel_name_none_default(self):
        m = parse_ncu_csv(MINIMAL_CSV)
        assert m.kernel_name == ""

    def test_alternate_column_names(self):
        """CSV with lowercase alternate column names parses correctly."""
        csv = (
            'metric_name,metric_value,metric_unit\n'
            'gpu__time_duration.sum,50000,nsecond\n'
            'sm__throughput.avg_pct_of_peak_sustained_elapsed,70.0,%\n'
        )
        m = parse_ncu_csv(csv, kernel_name="alt")
        assert m.kernel_name == "alt"
        assert m.gpu_time_us == pytest.approx(50.0)
        assert m.sm_throughput == pytest.approx(70.0)


# ── Dict Parser Tests ─────────────────────────────────────────

class TestParseNCUDict:

    def test_valid_dict(self):
        metrics = {
            "gpu__time_duration.sum": 50000.0,
            "sm__throughput.avg_pct_of_peak_sustained_elapsed": 88.0,
            "dram__throughput.avg_pct_of_peak_sustained_elapsed": 30.0,
            "sm__warps_active.avg.pct_of_peak_sustained_active": 80.0,
            "launch__registers_per_thread": 24,
        }
        m = parse_ncu_dict(metrics, kernel_name="test")
        assert m.gpu_time_us == pytest.approx(50.0)
        assert m.sm_throughput == pytest.approx(88.0)
        assert m.achieved_occupancy == pytest.approx(0.80)
        assert m.registers_per_thread == 24

    def test_empty_dict(self):
        m = parse_ncu_dict({}, kernel_name="empty")
        assert m.kernel_name == "empty"
        assert m.gpu_time_us is None
        assert m.sm_throughput is None

    def test_occupancy_as_percentage(self):
        metrics = {
            "sm__warps_active.avg.pct_of_peak_sustained_active": 62.5,
        }
        m = parse_ncu_dict(metrics)
        assert m.achieved_occupancy == pytest.approx(0.625)

    def test_gpu_name_from_dict(self):
        metrics = {"gpu_name": "NVIDIA H100"}
        m = parse_ncu_dict(metrics)
        assert m.gpu_name == "NVIDIA H100"

    def test_unknown_keys_stored_in_raw(self):
        metrics = {
            "gpu__time_duration.sum": 1000.0,
            "custom_unknown_metric": 42.0,
        }
        m = parse_ncu_dict(metrics)
        assert m.raw_metrics is not None
        assert m.raw_metrics["custom_unknown_metric"] == 42.0

    def test_string_values_preserved(self):
        metrics = {"some_metric": "not_a_number"}
        m = parse_ncu_dict(metrics)
        assert m.raw_metrics is not None
        assert m.raw_metrics["some_metric"] == "not_a_number"

    def test_launch_dims_from_dict(self):
        metrics = {
            "launch__block_size": [16, 16, 1],
            "launch__grid_size": [64, 64, 1],
        }
        m = parse_ncu_dict(metrics)
        assert m.block_size == (16, 16, 1)
        assert m.grid_size == (64, 64, 1)

    def test_individual_dims_from_dict(self):
        metrics = {
            "launch__grid_size_x": 128,
            "launch__grid_size_y": 256,
            "launch__grid_size_z": 1,
        }
        m = parse_ncu_dict(metrics)
        assert m.grid_size == (128, 256, 1)
