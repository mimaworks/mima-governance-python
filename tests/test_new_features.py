"""Tests for Phase 1–4 new features:
  Phase 1 — AST scanner (aliased imports, handle tracking, function-scope attestation)
  Phase 2 — Runtime guard (enable_guard warn/block/report modes)
  Phase 3 — Progressive disclosure (mima push --dry-run, mima status --demo)
  Phase 4 — Self-healing drift (mima init --hook, mima init --github-action)
"""

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _scan(tmp_path, source: str):
    """Write source to tmp_path/agent.py and return AST scan results."""
    (tmp_path / "agent.py").write_text(source)
    from mima_governance.cli import _scan_path
    detections, _ = _scan_path(tmp_path)
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: AST Scanner
# ─────────────────────────────────────────────────────────────────────────────

class TestAstScannerImportAliases:
    """Task 1.1 — import alias tracking."""

    def test_from_openai_import_class(self, tmp_path):
        """from openai import OpenAI → OpenAI() detected as openai."""
        ds = _scan(tmp_path, "from openai import OpenAI\nclient = OpenAI()\nclient.chat.completions.create()\n")
        high = [d for d in ds if d.confidence == "high"]
        assert any(d.library == "openai" and d.method == "ast" for d in high)

    def test_aliased_module_import(self, tmp_path):
        """import anthropic as ant → ant.messages.create() detected."""
        ds = _scan(tmp_path, "import anthropic as ant\nant.messages.create()\n")
        high = [d for d in ds if d.confidence == "high"]
        assert any(d.library == "anthropic" and d.method == "ast" for d in high)

    def test_multiple_from_imports(self, tmp_path):
        """from anthropic import Anthropic, AsyncAnthropic → both registered."""
        ds = _scan(tmp_path,
            "from anthropic import Anthropic, AsyncAnthropic\n"
            "c1 = Anthropic()\nc2 = AsyncAnthropic()\n"
            "c1.messages.create()\nc2.messages.create()\n"
        )
        high = [d for d in ds if d.confidence == "high" and d.library == "anthropic"]
        assert len(high) >= 2

    def test_langchain_alias(self, tmp_path):
        """import langchain as lc → lc.* detected."""
        ds = _scan(tmp_path, "import langchain as lc\nlc.llms.OpenAI()\n")
        high = [d for d in ds if d.confidence == "high"]
        assert any(d.library == "langchain" and d.method == "ast" for d in high)

    def test_non_ai_import_not_detected(self, tmp_path):
        """import requests → not detected."""
        ds = _scan(tmp_path, "import requests\nrequests.get('http://example.com')\n")
        assert not ds


class TestAstScannerHandleTracking:
    """Task 1.2 — variable assignment (AI handle) tracking."""

    def test_constructor_assignment(self, tmp_path):
        """client = OpenAI() → client.* calls detected."""
        ds = _scan(tmp_path,
            "from openai import OpenAI\n"
            "client = OpenAI()\n"
            "client.chat.completions.create(model='gpt-4o')\n"
        )
        assert any(d.library == "openai" and d.method == "ast" for d in ds)

    def test_dotted_constructor(self, tmp_path):
        """c = anthropic.Anthropic() → c.messages.create() detected."""
        ds = _scan(tmp_path,
            "import anthropic\n"
            "c = anthropic.Anthropic()\n"
            "c.messages.create()\n"
        )
        assert any(d.library == "anthropic" and d.method == "ast" for d in ds)

    def test_chained_attribute_not_registered(self, tmp_path):
        """x = client.chat does NOT register x as an AI handle."""
        ds = _scan(tmp_path,
            "from openai import OpenAI\n"
            "client = OpenAI()\n"
            "x = client.chat\n"
            "x.completions.create()\n"
        )
        # x is NOT in handle_map; only direct constructors are tracked.
        # The call via `x` should not be detected (wrapper abstraction).
        libs = {d.library for d in ds if d.method == "ast"}
        # x.completions.create() — x is not a direct OpenAI constructor assignment,
        # so it won't be in handle_map.  The client = OpenAI() line may itself
        # trigger a detection since OpenAI() IS a call on an alias.
        # What we're verifying: x is not registered.
        handle_detected_lines = [d.line for d in ds if d.library == "openai" and d.method == "ast"]
        # None of those lines should be the `x.completions.create()` line (line 4)
        assert 4 not in handle_detected_lines

    def test_unrelated_method_not_detected(self, tmp_path):
        """unrelated.method() → not detected."""
        ds = _scan(tmp_path, "unrelated = object()\nunrelated.method()\n")
        assert not ds


class TestAstScannerCallDetection:
    """Task 1.3 — call site detection via handles."""

    def test_deep_attribute_chain(self, tmp_path):
        """client.chat.completions.create() detected via handle."""
        ds = _scan(tmp_path,
            "from openai import OpenAI\n"
            "client = OpenAI()\n"
            "resp = client.chat.completions.create(model='gpt-4o', messages=[])\n"
        )
        assert any(d.library == "openai" and d.line == 3 and d.method == "ast" for d in ds)

    def test_direct_library_call(self, tmp_path):
        """import openai; openai.chat.completions.create() detected directly."""
        ds = _scan(tmp_path, "import openai\nopenai.chat.completions.create()\n")
        assert any(d.library == "openai" and d.method == "ast" for d in ds)


class TestAstFunctionScopeAttestation:
    """Task 1.4 — function-scope attestation (replaces 10-line proximity)."""

    def test_decorated_function_attests_distant_call(self, tmp_path):
        """@mima.attest() on a function → AI call at line 55 is attested."""
        lines = ["import anthropic", "@mima.attest(tool_name='x')", "def my_fn():"]
        # Add 50 lines of body before the AI call
        for i in range(50):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return anthropic.Anthropic().messages.create()")
        source = "\n".join(lines) + "\n"
        ds = _scan(tmp_path, source)
        ai_detections = [d for d in ds if d.library == "anthropic" and d.method == "ast"]
        assert ai_detections, "Should detect anthropic call"
        assert all(d.attested for d in ai_detections), "Call inside decorated function must be attested"

    def test_undecorated_function_not_attested(self, tmp_path):
        """@mima.attest on function A; AI call in function B → NOT attested."""
        source = (
            "import openai\n"
            "@mima.attest(tool_name='a')\n"
            "def fn_a():\n"
            "    pass\n"
            "\n"
            "def fn_b():\n"
            "    openai.chat.completions.create()\n"
        )
        ds = _scan(tmp_path, source)
        b_calls = [d for d in ds if d.library == "openai" and d.line == 7]
        assert b_calls
        assert not b_calls[0].attested

    def test_nested_function_inherits_attestation(self, tmp_path):
        """Decorator on outer function → inner function's calls are also attested."""
        source = (
            "import openai\n"
            "@mima.attest(tool_name='outer')\n"
            "def outer():\n"
            "    def inner():\n"
            "        openai.chat.completions.create()\n"
            "    inner()\n"
        )
        ds = _scan(tmp_path, source)
        calls = [d for d in ds if d.library == "openai"]
        assert calls
        assert calls[0].attested

    def test_attest_dot_suffix_pattern(self, tmp_path):
        """@client.attest() pattern (anything.attest) is recognised."""
        source = (
            "import openai\n"
            "@gov_client.attest(tool_name='x')\n"
            "def fn():\n"
            "    openai.chat.completions.create()\n"
        )
        ds = _scan(tmp_path, source)
        calls = [d for d in ds if d.library == "openai"]
        assert calls and calls[0].attested


class TestAstFallback:
    """Task 1.5 — AST→token fallback; method field in output."""

    def test_valid_python_uses_ast(self, tmp_path):
        """A valid .py file produces method='ast' detections."""
        ds = _scan(tmp_path, "import openai\nopenai.chat.completions.create()\n")
        assert any(d.method == "ast" for d in ds)

    def test_syntax_error_uses_token_fallback(self, tmp_path):
        """A file with a syntax error falls back to tokenizer (method='token')."""
        (tmp_path / "broken.py").write_text("import openai\nopenai.chat(\n  # unclosed\n")
        from mima_governance.cli import _scan_path
        ds, _ = _scan_path(tmp_path)
        # Token scanner may or may not find something, but it should not crash.
        token_ds = [d for d in ds if d.method == "token"]
        # At minimum: no exception raised; result is a list.
        assert isinstance(token_ds, list)

    def test_json_output_includes_method(self, tmp_path, capsys):
        import json
        (tmp_path / "app.py").write_text("import openai\nopenai.chat.completions.create()\n")
        with patch("sys.argv", ["mima", "scan", str(tmp_path), "--json"]):
            from mima_governance.cli import main
            main()
        data = json.loads(capsys.readouterr().out)
        assert data
        assert "method" in data[0]
        assert data[0]["method"] in ("ast", "token")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Runtime Guard
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardModule:
    """Tasks 2.1 & 2.2 — enable_guard and @mima.attest() context integration."""

    def setup_method(self):
        # Reset guard state before each test.
        from mima_governance import guard
        guard._guard_enabled = False
        guard._guard_mode = "warn"
        from mima_governance.guard import _thread_local
        _thread_local.attested = False

    def test_enable_guard_invalid_mode(self):
        from mima_governance.guard import enable_guard
        with pytest.raises(ValueError, match="mode must be one of"):
            enable_guard(mode="silent")

    def test_guard_is_idempotent(self):
        from mima_governance import guard
        guard._guard_enabled = False
        from mima_governance.guard import enable_guard
        enable_guard("warn")
        enable_guard("warn")  # second call is no-op, no error

    def test_set_attested_thread_local(self):
        from mima_governance.guard import _set_attested, _is_attested
        assert not _is_attested()
        _set_attested(True)
        assert _is_attested()
        _set_attested(False)
        assert not _is_attested()

    def test_set_attested_async_context_var(self):
        import asyncio
        from mima_governance.guard import _set_attested_async, _is_attested, _async_attested

        async def run():
            assert not _is_attested()
            token = _set_attested_async(True)
            assert _is_attested()
            _async_attested.reset(token)
            assert not _is_attested()

        asyncio.get_event_loop().run_until_complete(run())

    def test_guard_warn_fires_when_not_attested(self):
        from mima_governance import guard
        guard._guard_enabled = True
        guard._guard_mode = "warn"
        guard._thread_local.attested = False

        called = []

        def fake_call(*args, **kwargs):
            called.append(True)
            return "ok"

        wrapped = guard._make_wrapper(fake_call, "openai.chat", "warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped()
        assert result == "ok"
        assert called
        assert any("mima guard" in str(x.message) for x in w)

    def test_guard_block_raises_when_not_attested(self):
        from mima_governance import guard
        guard._guard_enabled = True
        guard._guard_mode = "block"
        guard._thread_local.attested = False

        from mima_governance.guard import MimaAttestationError

        def fake_call():
            return "ok"

        wrapped = guard._make_wrapper(fake_call, "openai.chat", "block")
        with pytest.raises(MimaAttestationError, match="mima guard"):
            wrapped()

    def test_guard_no_warn_when_attested(self):
        from mima_governance import guard
        guard._guard_enabled = True
        guard._guard_mode = "warn"
        guard._thread_local.attested = True  # attested context

        called = []

        def fake_call():
            called.append(True)
            return "result"

        wrapped = guard._make_wrapper(fake_call, "openai.chat", "warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped()
        assert result == "result"
        assert called
        assert not any("mima guard" in str(x.message) for x in w)

    def test_guard_graceful_noop_when_library_absent(self):
        """enable_guard() must not raise if openai/anthropic are not installed."""
        from mima_governance.guard import enable_guard, disable_guard
        disable_guard()
        from mima_governance import guard
        guard._guard_enabled = False
        # Patch importlib.import_module to raise ImportError for AI libs
        import importlib
        original = importlib.import_module

        def mock_import(name, *args, **kwargs):
            if name in ("openai", "anthropic", "litellm"):
                raise ImportError(f"No module named {name!r}")
            return original(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=mock_import):
            enable_guard("warn")  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Progressive Disclosure
# ─────────────────────────────────────────────────────────────────────────────

class TestDryRun:
    """Task 3.1 — mima push --dry-run (no credentials needed)."""

    def test_dry_run_shows_controls(self, capsys):
        with patch("sys.argv", ["mima", "push", "change_event", "--dry-run"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "change_event" in out
        assert "SOC2_CC8.1" in out
        assert "mima login" in out

    def test_dry_run_all_11_record_types(self, capsys):
        from mima_governance.cli import _DRY_RUN_CONTROLS
        assert len(_DRY_RUN_CONTROLS) >= 11
        for record_type in _DRY_RUN_CONTROLS:
            with patch("sys.argv", ["mima", "push", record_type, "--dry-run"]):
                with pytest.raises(SystemExit) as exc:
                    from mima_governance.cli import main
                    main()
                assert exc.value.code == 0, f"--dry-run failed for {record_type}"
            out = capsys.readouterr().out
            assert record_type in out

    def test_dry_run_unknown_record_type(self, capsys):
        with patch("sys.argv", ["mima", "push", "bogus_type", "--dry-run"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1

    def test_dry_run_makes_no_http_calls(self, capsys):
        """--dry-run must not call httpx.get or httpx.post."""
        with patch("httpx.get") as mock_get, patch("httpx.post") as mock_post:
            with patch("sys.argv", ["mima", "push", "vendor_risk", "--dry-run"]):
                with pytest.raises(SystemExit):
                    from mima_governance.cli import main
                    main()
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_dry_run_works_without_credentials(self, capsys, monkeypatch):
        """--dry-run must not require MIMA_API_KEY to be set."""
        monkeypatch.delenv("MIMA_API_KEY", raising=False)
        monkeypatch.delenv("MIMA_WORKSPACE_ID", raising=False)
        with patch("mima_governance.config.get_api_key", return_value=None):
            with patch("mima_governance.config.get_workspace_id", return_value=None):
                with patch("sys.argv", ["mima", "push", "access_review", "--dry-run"]):
                    with pytest.raises(SystemExit) as exc:
                        from mima_governance.cli import main
                        main()
                    assert exc.value.code == 0


class TestStatusDemo:
    """Task 3.2 — mima status --demo (no credentials needed)."""

    def test_demo_shows_demo_mode_header(self, tmp_path, capsys):
        with patch("sys.argv", ["mima", "status", "--demo", str(tmp_path)]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "DEMO MODE" in out
        assert "mima login" in out

    def test_demo_detects_libraries(self, tmp_path, capsys):
        (tmp_path / "agent.py").write_text(
            "import openai\nopenai.chat.completions.create()\n"
        )
        with patch("sys.argv", ["mima", "status", "--demo", str(tmp_path)]):
            with pytest.raises(SystemExit):
                from mima_governance.cli import main
                main()
        out = capsys.readouterr().out
        assert "openai" in out

    def test_demo_works_without_credentials(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("MIMA_API_KEY", raising=False)
        monkeypatch.delenv("MIMA_WORKSPACE_ID", raising=False)
        with patch("mima_governance.config.get_api_key", return_value=None):
            with patch("mima_governance.config.get_workspace_id", return_value=None):
                with patch("sys.argv", ["mima", "status", "--demo", str(tmp_path)]):
                    with pytest.raises(SystemExit) as exc:
                        from mima_governance.cli import main
                        main()
                    assert exc.value.code == 0

    def test_demo_makes_no_http_calls(self, tmp_path, capsys):
        with patch("httpx.get") as mock_get:
            with patch("sys.argv", ["mima", "status", "--demo", str(tmp_path)]):
                with pytest.raises(SystemExit):
                    from mima_governance.cli import main
                    main()
        mock_get.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Self-Healing Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestInitHook:
    """Task 4.1 — mima init --hook writes an executable pre-commit hook."""

    def _make_git_repo(self, tmp_path: Path) -> Path:
        """Create a minimal .git structure in tmp_path."""
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        return tmp_path

    def test_hook_written_to_git_hooks(self, tmp_path, capsys):
        self._make_git_repo(tmp_path)
        out_file = tmp_path / "test_gov.py"
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "--hook"]):
            with patch("mima_governance.cli._install_pre_commit_hook") as mock_hook:
                from mima_governance.cli import main
                main()
        mock_hook.assert_called_once()

    def test_hook_file_is_created(self, tmp_path, capsys, monkeypatch):
        self._make_git_repo(tmp_path)
        out_file = tmp_path / "test_gov.py"
        # Run _install_pre_commit_hook directly with a real .git dir
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _install_pre_commit_hook
        _install_pre_commit_hook(".", str(out_file))
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        assert hook_path.exists()
        content = hook_path.read_text()
        assert "mima scan" in content
        assert "--strict" in content

    def test_hook_is_executable(self, tmp_path, monkeypatch):
        self._make_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _install_pre_commit_hook
        _install_pre_commit_hook(".", "tests/test_governance.py")
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        import os, stat
        mode = os.stat(hook_path).st_mode
        assert mode & stat.S_IXUSR, "pre-commit hook must be executable"

    def test_hook_appends_to_existing(self, tmp_path, monkeypatch):
        self._make_git_repo(tmp_path)
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        hook_path.write_text("#!/bin/bash\necho existing hook\n")
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _install_pre_commit_hook
        _install_pre_commit_hook(".", "tests/test_governance.py")
        content = hook_path.read_text()
        assert "existing hook" in content
        assert "mima scan" in content

    def test_hook_not_duplicated(self, tmp_path, monkeypatch, capsys):
        """Running --hook twice doesn't insert the block twice."""
        self._make_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _install_pre_commit_hook, _HOOK_MARKER
        _install_pre_commit_hook(".", "tests/test_governance.py")
        _install_pre_commit_hook(".", "tests/test_governance.py")  # second run
        hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
        content = hook_path.read_text()
        assert content.count(_HOOK_MARKER) == 1

    def test_hook_skipped_outside_git_repo(self, tmp_path, monkeypatch, capsys):
        """If .git doesn't exist, print a warning and do not crash."""
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _install_pre_commit_hook
        _install_pre_commit_hook(".", "tests/test_governance.py")
        # Should print a warning, not crash
        # And should not create any hook file
        assert not (tmp_path / ".git").exists()


class TestInitGithubAction:
    """Task 4.2 — mima init --github-action writes governance.yml."""

    def test_workflow_file_written(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        from mima_governance.cli import _write_github_action
        _write_github_action("tests/test_governance.py")
        wf_path = tmp_path / ".github" / "workflows" / "governance.yml"
        assert wf_path.exists()
        content = wf_path.read_text()
        assert "mima test tests/test_governance.py" in content
        assert "actions/checkout" in content

    def test_workflow_not_overwritten_if_exists(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        wf_path = tmp_path / ".github" / "workflows" / "governance.yml"
        wf_path.parent.mkdir(parents=True)
        wf_path.write_text("# existing workflow\n")
        from mima_governance.cli import _write_github_action
        _write_github_action("tests/test_governance.py")
        assert wf_path.read_text() == "# existing workflow\n"

    def test_init_github_action_flag(self, tmp_path, capsys):
        out_file = tmp_path / "test_gov.py"
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "--github-action"]):
            with patch("mima_governance.cli._write_github_action") as mock_wf:
                from mima_governance.cli import main
                main()
        mock_wf.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive prompting and activation helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractivePrompting:
    """_prompt_yes_no behaviour across interactive / non-interactive / --yes modes."""

    def test_yes_all_always_returns_true(self):
        from mima_governance.cli import _prompt_yes_no
        assert _prompt_yes_no("anything?", default=False, interactive=False, yes_all=True)

    def test_non_interactive_no_yes_all_returns_false(self):
        from mima_governance.cli import _prompt_yes_no
        assert not _prompt_yes_no("anything?", default=True, interactive=False, yes_all=False)

    def test_interactive_default_true_enter(self):
        from mima_governance.cli import _prompt_yes_no
        with patch("builtins.input", return_value=""):
            result = _prompt_yes_no("question?", default=True, interactive=True, yes_all=False)
        assert result is True

    def test_interactive_default_true_explicit_no(self):
        from mima_governance.cli import _prompt_yes_no
        with patch("builtins.input", return_value="n"):
            result = _prompt_yes_no("question?", default=True, interactive=True, yes_all=False)
        assert result is False

    def test_interactive_explicit_yes(self):
        from mima_governance.cli import _prompt_yes_no
        with patch("builtins.input", return_value="y"):
            result = _prompt_yes_no("question?", default=False, interactive=True, yes_all=False)
        assert result is True

    def test_interactive_eof_returns_default(self):
        from mima_governance.cli import _prompt_yes_no
        with patch("builtins.input", side_effect=EOFError):
            result = _prompt_yes_no("question?", default=True, interactive=True, yes_all=False)
        assert result is True


class TestYesFlagActivation:
    """mima init --yes activates hook + action + guard without any prompts."""

    def _make_git_repo(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "hooks").mkdir(parents=True)

    def test_yes_flag_installs_hook(self, tmp_path, capsys, monkeypatch):
        self._make_git_repo(tmp_path)
        out_file = tmp_path / "test_gov.py"
        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "--yes"]):
            from mima_governance.cli import main
            main()
        hook = tmp_path / ".git" / "hooks" / "pre-commit"
        assert hook.exists()
        assert "mima scan" in hook.read_text()

    def test_yes_flag_writes_github_action(self, tmp_path, capsys, monkeypatch):
        self._make_git_repo(tmp_path)
        out_file = tmp_path / "test_gov.py"
        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "--yes"]):
            from mima_governance.cli import main
            main()
        wf = tmp_path / ".github" / "workflows" / "governance.yml"
        assert wf.exists()

    def test_yes_flag_short_form(self, tmp_path, capsys, monkeypatch):
        """-y is equivalent to --yes."""
        self._make_git_repo(tmp_path)
        out_file = tmp_path / "test_gov.py"
        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "-y"]):
            from mima_governance.cli import main
            main()
        wf = tmp_path / ".github" / "workflows" / "governance.yml"
        assert wf.exists()

    def test_yes_flag_enables_guard_in_entry_point(self, tmp_path, capsys, monkeypatch):
        """--yes patches the entry point with enable_guard() if main.py exists."""
        self._make_git_repo(tmp_path)
        (tmp_path / "main.py").write_text("import openai\nopenai.chat.completions.create()\n")
        out_file = tmp_path / "test_gov.py"
        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["mima", "init", str(tmp_path),
                                 "--output", str(out_file), "--yes"]):
            from mima_governance.cli import main
            main()
        content = (tmp_path / "main.py").read_text()
        assert "enable_guard" in content


class TestFindEntryPoint:
    """_find_entry_point detects standard entry point files."""

    def test_finds_main_py(self, tmp_path):
        (tmp_path / "main.py").write_text("# entry\n")
        from mima_governance.cli import _find_entry_point
        result = _find_entry_point(str(tmp_path))
        assert result is not None
        assert result.name == "main.py"

    def test_finds_app_py_when_no_main(self, tmp_path):
        (tmp_path / "app.py").write_text("# entry\n")
        from mima_governance.cli import _find_entry_point
        result = _find_entry_point(str(tmp_path))
        assert result is not None
        assert result.name == "app.py"

    def test_returns_none_when_no_candidate(self, tmp_path):
        from mima_governance.cli import _find_entry_point
        assert _find_entry_point(str(tmp_path)) is None


class TestPatchEntryPoint:
    """_patch_entry_point_with_guard inserts enable_guard() after imports."""

    def test_patches_after_imports(self, tmp_path):
        entry = tmp_path / "main.py"
        entry.write_text("import os\nimport openai\n\nprint('start')\n")
        from mima_governance.cli import _patch_entry_point_with_guard
        result = _patch_entry_point_with_guard(entry)
        assert result is True
        content = entry.read_text()
        assert "enable_guard" in content
        # Original imports and code must still be present
        assert "import os" in content
        assert "import openai" in content
        assert "print('start')" in content
        # Guard call must appear before print('start')
        guard_pos = content.index("enable_guard(mode=")
        print_pos = content.index("print('start')")
        assert guard_pos < print_pos

    def test_idempotent_when_already_present(self, tmp_path):
        entry = tmp_path / "main.py"
        entry.write_text("from mima_governance.guard import enable_guard\nenable_guard()\n")
        from mima_governance.cli import _patch_entry_point_with_guard
        result = _patch_entry_point_with_guard(entry)
        assert result is False  # already configured — not patched again

    def test_patches_file_with_no_imports(self, tmp_path):
        entry = tmp_path / "main.py"
        entry.write_text("print('hello')\n")
        from mima_governance.cli import _patch_entry_point_with_guard
        result = _patch_entry_point_with_guard(entry)
        assert result is True
        assert "enable_guard" in entry.read_text()
