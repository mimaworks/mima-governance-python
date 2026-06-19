"""Tests for the extended CLI: login, status, test, scan."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestMainDispatch:
    """Test the top-level mima CLI dispatch."""

    def test_help_flag(self, capsys):
        with patch("sys.argv", ["mima", "--help"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "mima scan" in out
        assert "mima init" in out
        assert "mima test" in out
        assert "mima status" in out
        assert "mima login" in out

    def test_version_flag(self, capsys):
        with patch("sys.argv", ["mima", "--version"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "0.3.0" in out

    def test_unknown_command(self, capsys):
        with patch("sys.argv", ["mima", "bogus"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "unknown command" in err


class TestCmdScan:
    """Test mima scan subcommand."""

    def test_scan_nonexistent_path(self):
        with patch("sys.argv", ["mima", "scan", "/nonexistent/path"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1

    def test_scan_empty_dir(self, tmp_path, capsys):
        with patch("sys.argv", ["mima", "scan", str(tmp_path)]):
            from mima_governance.cli import main
            main()
        out = capsys.readouterr().out
        assert "No AI library call sites found" in out

    def test_scan_detects_openai(self, tmp_path, capsys):
        test_file = tmp_path / "app.py"
        test_file.write_text("import openai\nresult = openai.chat.completions.create()\n")
        with patch("sys.argv", ["mima", "scan", str(tmp_path)]):
            from mima_governance.cli import main
            main()
        out = capsys.readouterr().out
        assert "openai" in out
        assert "unattested" in out

    def test_scan_json_output(self, tmp_path, capsys):
        test_file = tmp_path / "app.py"
        test_file.write_text("import anthropic\nx = anthropic.Client()\n")
        with patch("sys.argv", ["mima", "scan", str(tmp_path), "--json"]):
            from mima_governance.cli import main
            main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["library"] == "anthropic"


class TestCmdInit:
    """Test mima init subcommand."""

    def test_init_help(self, capsys):
        with patch("sys.argv", ["mima", "init", "--help"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "test_governance.py" in out
        assert "--output" in out

    def test_init_generates_file(self, tmp_path, capsys):
        out_file = tmp_path / "tests" / "test_gov.py"
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file)]):
            from mima_governance.cli import main
            main()
        assert out_file.exists()
        content = out_file.read_text()
        assert "GovernanceTest" in content
        assert "test_all_calls_attested" in content
        assert "test_coverage_threshold" in content

    def test_init_detects_openai(self, tmp_path, capsys):
        (tmp_path / "agent.py").write_text(
            "import openai\nopenai.chat.completions.create()\n"
        )
        out_file = tmp_path / "test_gov.py"
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file)]):
            from mima_governance.cli import main
            main()
        content = out_file.read_text()
        assert "openai" in content
        out = capsys.readouterr().out
        assert "1 call" in out

    def test_init_refuses_overwrite_without_force(self, tmp_path):
        out_file = tmp_path / "test_gov.py"
        out_file.write_text("# existing\n")
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file)]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1

    def test_init_force_overwrites(self, tmp_path):
        out_file = tmp_path / "test_gov.py"
        out_file.write_text("# existing\n")
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "--force"]):
            from mima_governance.cli import main
            main()
        content = out_file.read_text()
        assert "GovernanceTest" in content

    def test_init_ci_snippet_in_output(self, tmp_path, capsys):
        out_file = tmp_path / "test_gov.py"
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file)]):
            from mima_governance.cli import main
            main()
        out = capsys.readouterr().out
        assert "pip install mima-governance" in out
        assert "mima login" in out


class TestCmdLogin:
    """Test mima login subcommand."""

    def test_login_help(self, capsys):
        with patch("sys.argv", ["mima", "login", "--help"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "authenticate" in out.lower()

    @patch("httpx.get")
    @patch("mima_governance.config.set_credentials")
    def test_login_success(self, mock_save, mock_get, capsys):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        with patch("sys.argv", [
            "mima", "login",
            "--api-key", "mima_ext_test123",
            "--workspace-id", "ws-abc",
            "--url", "http://localhost:8081",
        ]):
            from mima_governance.cli import main
            main()

        mock_save.assert_called_once_with("mima_ext_test123", "ws-abc", "http://localhost:8081")
        out = capsys.readouterr().out
        assert "Verifying credentials" in out
        assert "ws-abc" in out

    @patch("httpx.get")
    def test_login_invalid_key(self, mock_get, capsys):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        with patch("sys.argv", [
            "mima", "login",
            "--api-key", "bad_key",
            "--workspace-id", "ws-abc",
        ]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "401" in err or "invalid" in err.lower()


class TestCmdStatus:
    """Test mima status subcommand."""

    def test_status_not_logged_in(self, capsys, monkeypatch):
        monkeypatch.delenv("MIMA_API_KEY", raising=False)
        monkeypatch.delenv("MIMA_WORKSPACE_ID", raising=False)

        with patch("mima_governance.config.get_api_key", return_value=None):
            with patch("mima_governance.config.get_workspace_id", return_value=None):
                with patch("sys.argv", ["mima", "status"]):
                    with pytest.raises(SystemExit) as exc:
                        from mima_governance.cli import main
                        main()
                    assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "not logged in" in err

    @patch("httpx.get")
    def test_status_renders_dashboard(self, mock_get, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "mima_ext_test")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws-123")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "overall_pct": 54,
            "frameworks": [
                {"framework": "soc2_type2", "score_pct": 68, "controls_covered": 8,
                 "controls_required": 12, "records_total": 42, "validated_at": None, "validated_by": None},
                {"framework": "iso_27001", "score_pct": 54, "controls_covered": 6,
                 "controls_required": 11, "records_total": 42, "validated_at": "2026-06-01T00:00:00Z", "validated_by": "admin@co.com"},
            ],
        }
        mock_get.return_value = mock_resp

        with patch("sys.argv", ["mima", "status"]):
            from mima_governance.cli import main
            main()

        out = capsys.readouterr().out
        assert "SOC 2 Type II" in out
        assert "68%" in out
        assert "54%" in out
        assert "Overall: 54%" in out
        assert "\u2713 validated" in out

    @patch("httpx.get")
    def test_status_json_output(self, mock_get, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "mima_ext_test")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws-123")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"overall_pct": 54, "frameworks": []}
        mock_get.return_value = mock_resp

        with patch("sys.argv", ["mima", "status", "--json"]):
            from mima_governance.cli import main
            main()

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["overall_pct"] == 54


class TestCmdTest:
    """Test mima test subcommand."""

    def test_test_help(self, capsys):
        with patch("sys.argv", ["mima", "test", "--help"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "GovernanceTest" in out

    def test_test_coverage_mode(self, tmp_path, capsys):
        # Create a file with an unattested AI call
        test_file = tmp_path / "agent.py"
        test_file.write_text("import openai\nopenai.chat.completions.create()\n")

        with patch("sys.argv", ["mima", "test", "--coverage", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1  # Not fully attested
        out = capsys.readouterr().out
        assert "Coverage" in out
        assert "0%" in out

    def test_test_runs_test_file(self, tmp_path, capsys):
        # Write a governance test file
        test_file = tmp_path / "test_gov.py"
        test_file.write_text("""
from mima_governance.testing import GovernanceTest, assert_attested

class TestEmpty(GovernanceTest):
    def test_empty_dir_is_covered(self):
        result = self.scan("{scan_dir}")
        return assert_attested(result, min_coverage=1.0)
""".format(scan_dir=str(tmp_path / "empty_src")))

        # Create an empty source dir (no AI calls = 100% coverage)
        (tmp_path / "empty_src").mkdir()

        with patch("sys.argv", ["mima", "test", str(test_file)]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0  # all tests passed

        out = capsys.readouterr().out
        assert "1 passed" in out or "\u2713" in out
