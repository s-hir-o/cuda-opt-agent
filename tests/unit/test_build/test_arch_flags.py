"""Tests for architecture flag utilities."""

import pytest

from cuda_opt_agent.build.arch_flags import (
    common_flags,
    cubin_arch_flag,
    executable_arch_flag,
    gpu_name,
    is_known_arch,
    list_archs,
    ptx_arch_flag,
    validate_arch,
)
from cuda_opt_agent.core.exceptions import ConfigurationError


class TestValidateArch:

    def test_valid_arch(self):
        assert validate_arch("sm_80") == "sm_80"
        assert validate_arch("sm_89") == "sm_89"
        assert validate_arch("sm_90") == "sm_90"

    def test_normalizes_case(self):
        assert validate_arch("SM_80") == "sm_80"
        assert validate_arch(" Sm_80 ") == "sm_80"

    def test_invalid_format_raises(self):
        with pytest.raises(ConfigurationError, match="Invalid architecture"):
            validate_arch("ampere")
        with pytest.raises(ConfigurationError):
            validate_arch("sm_abc")
        with pytest.raises(ConfigurationError):
            validate_arch("")


class TestKnownArch:

    def test_known(self):
        assert is_known_arch("sm_80") is True
        assert is_known_arch("sm_90") is True

    def test_unknown_valid_format(self):
        assert is_known_arch("sm_50") is False

    def test_gpu_names(self):
        assert "A100" in gpu_name("sm_80")
        assert "H100" in gpu_name("sm_90")
        assert gpu_name("sm_50") == "Unknown GPU"

    def test_list_archs(self):
        archs = list_archs()
        assert "sm_80" in archs
        assert "sm_90" in archs
        assert len(archs) >= 4


class TestArchFlags:

    def test_ptx_flag(self):
        flags = ptx_arch_flag("sm_80")
        assert "-ptx" in flags
        assert "-arch=sm_80" in flags

    def test_cubin_flag(self):
        flags = cubin_arch_flag("sm_89")
        assert "-cubin" in flags
        assert "-arch=sm_89" in flags

    def test_executable_flag(self):
        flags = executable_arch_flag("sm_90")
        assert "-arch=sm_90" in flags
        assert "-cubin" not in flags
        assert "-ptx" not in flags

    def test_ptx_invalid_arch_raises(self):
        with pytest.raises(ConfigurationError):
            ptx_arch_flag("invalid")


class TestCommonFlags:

    def test_defaults(self):
        flags = common_flags()
        assert "-O3" in flags
        assert "-lineinfo" in flags
        assert "--use_fast_math" not in flags

    def test_fast_math(self):
        flags = common_flags(fast_math=True)
        assert "--use_fast_math" in flags

    def test_no_lineinfo(self):
        flags = common_flags(lineinfo=False)
        assert "-lineinfo" not in flags

    def test_custom_optimization(self):
        flags = common_flags(optimization_level=0)
        assert "-O0" in flags

    def test_extra_flags(self):
        flags = common_flags(extra=["-DNDEBUG", "-I/usr/local/include"])
        assert "-DNDEBUG" in flags
        assert "-I/usr/local/include" in flags

    def test_all_valid_levels(self):
        for level in (0, 1, 2, 3):
            flags = common_flags(optimization_level=level)
            assert f"-O{level}" in flags

    def test_invalid_optimization_level_raises(self):
        with pytest.raises(ConfigurationError, match="Invalid optimization level"):
            common_flags(optimization_level=4)
        with pytest.raises(ConfigurationError):
            common_flags(optimization_level=-1)
        with pytest.raises(ConfigurationError):
            common_flags(optimization_level=7)
