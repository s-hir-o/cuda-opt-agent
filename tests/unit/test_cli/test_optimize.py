"""Tests for optimize CLI command."""

import pytest
from typer.testing import CliRunner

from cuda_opt_agent.cli.app import app

runner = CliRunner()


class TestOptimizeCommand:

    def test_optimize_missing_file(self):
        result = runner.invoke(app, ["optimize", "/nonexistent/file.cu"])
        assert result.exit_code == 1

    def test_optimize_help(self):
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "CUDA source file" in result.output
