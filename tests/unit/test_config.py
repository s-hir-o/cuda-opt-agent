"""Tests for core configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cuda_opt_agent.core.config import AppSettings, CompilerConfig, LLMConfig, load_settings


class TestAppSettingsDefaults:
    """Verify all default values are correct."""

    def test_default_project_name(self):
        settings = AppSettings()
        assert settings.project.name == "cuda-opt-agent"

    def test_default_arch(self):
        settings = AppSettings()
        assert settings.compiler.default_arch == "sm_80"

    def test_default_llm_provider(self):
        settings = AppSettings()
        assert settings.llm.provider == "claude"

    def test_default_max_iterations(self):
        settings = AppSettings()
        assert settings.optimization.max_iterations == 20

    def test_default_benchmark_iterations(self):
        settings = AppSettings()
        assert settings.benchmark.warmup_iterations == 3
        assert settings.benchmark.timing_iterations == 10

    def test_default_search_strategy(self):
        settings = AppSettings()
        assert settings.search.strategy == "grid"

    def test_default_ptx_enabled(self):
        settings = AppSettings()
        assert settings.analysis.ptx_enabled is True
        assert settings.compiler.ptx_enabled is True

    def test_default_database_url(self):
        settings = AppSettings()
        assert "sqlite" in settings.database.url


class TestLoadSettings:
    """Test YAML config loading."""

    def test_load_from_yaml(self, tmp_path: Path):
        config = textwrap.dedent("""\
            compiler:
              default_arch: "sm_89"
              ptx_enabled: false
            llm:
              provider: "openai"
            optimization:
              max_iterations: 50
        """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config)

        settings = load_settings(config_file)
        assert settings.compiler.default_arch == "sm_89"
        assert settings.compiler.ptx_enabled is False
        assert settings.llm.provider == "openai"
        assert settings.optimization.max_iterations == 50

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        settings = load_settings(tmp_path / "nonexistent.yaml")
        assert settings.compiler.default_arch == "sm_80"

    def test_load_none_returns_defaults(self):
        settings = load_settings(None)
        assert settings.llm.provider == "claude"

    def test_load_empty_yaml_returns_defaults(self, tmp_path: Path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        settings = load_settings(config_file)
        assert settings.project.name == "cuda-opt-agent"


class TestNestedConfig:
    """Test nested config models independently."""

    def test_compiler_config_extra_flags(self):
        config = CompilerConfig(extra_flags=["-O3", "--use_fast_math"])
        assert "-O3" in config.extra_flags
        assert len(config.extra_flags) == 2

    def test_llm_config_claude_model(self):
        config = LLMConfig()
        assert config.claude.model == "claude-sonnet-4-20250514"

    def test_llm_config_openai_model(self):
        config = LLMConfig()
        assert config.openai.model == "gpt-4o"
