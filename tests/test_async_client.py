"""Tests for AsyncMimaGovernance — mirrors test_grc.py structure with pytest-asyncio."""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_attest_resp(attestation_id="attest-async-1"):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "attestation_id": attestation_id,
        "external_verified": False,
        "trust_tier": "declared",
        "detail": "ok",
    }
    return mock


def _mock_grc_resp(record_id="rec-async-1", record_type="access_review", controls=None):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "record_id": record_id,
        "record_type": record_type,
        "mapped_controls": controls or [],
    }
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def async_client():
    from mima_governance import AsyncMimaGovernance
    return AsyncMimaGovernance(
        workspace_id="ws-async-test",
        api_key="mima_ext_async",
        system_name="async-pipeline",
        base_url="http://localhost:8081",
    )


# ---------------------------------------------------------------------------
# Import + context manager
# ---------------------------------------------------------------------------

class TestAsyncImport:
    def test_import_works(self):
        from mima_governance import AsyncMimaGovernance
        assert AsyncMimaGovernance is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes(self):
        from mima_governance import AsyncMimaGovernance
        with patch("httpx.AsyncClient") as MockClient:
            instance = MagicMock()
            instance.aclose = AsyncMock()
            instance.post = AsyncMock(return_value=_mock_grc_resp())
            MockClient.return_value = instance
            async with AsyncMimaGovernance(
                workspace_id="ws", api_key="key", system_name="sys",
                base_url="http://localhost:8081",
            ) as mima:
                pass
            instance.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_close_flushes_and_closes(self, async_client):
        async_client._http.aclose = AsyncMock()
        async_client._http.post = AsyncMock(return_value=_mock_grc_resp())
        await async_client.close()
        async_client._http.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# Async GRC methods
# ---------------------------------------------------------------------------

class TestAsyncGrcMethods:
    @pytest.mark.asyncio
    async def test_access_review_returns_grc_result(self, async_client):
        async_client._http.post = AsyncMock(
            return_value=_mock_grc_resp("rec-ar-1", "access_review", ["SOC2_CC6.1"])
        )
        result = await async_client.access_review(
            "alice@co.com", "crm-system", True, reviewed_by="bob@co.com"
        )
        assert result.record_id == "rec-ar-1"
        assert "SOC2_CC6.1" in result.mapped_controls

    @pytest.mark.asyncio
    async def test_change_event_returns_grc_result(self, async_client):
        async_client._http.post = AsyncMock(
            return_value=_mock_grc_resp("rec-ce-1", "change_event", ["SOC2_CC8.1"])
        )
        result = await async_client.change_event(
            "deployment", "ci-bot@co.com", "Deploy v2.0",
            environment="production", system="api",
        )
        assert result.record_id == "rec-ce-1"

    @pytest.mark.asyncio
    async def test_ai_risk_assessment_async(self, async_client):
        async_client._http.post = AsyncMock(
            return_value=_mock_grc_resp("rec-ara-1", "ai_risk_assessment", ["EUAIA_ART9"])
        )
        result = await async_client.ai_risk_assessment(
            "loan-model", "high", "credit_scoring",
            impact_domains=["credit"],
            art5_self_assessment=True,
            assessor="j.smith@co.com",
        )
        assert result.record_id == "rec-ara-1"

    @pytest.mark.asyncio
    async def test_governance_review_async(self, async_client):
        async_client._http.post = AsyncMock(
            return_value=_mock_grc_resp("rec-gr-1", "governance_review", ["SOC2_CC2.1"])
        )
        result = await async_client.governance_review(
            "board-audit", "q1_review",
            frameworks_reviewed=["soc2_type2"],
            overall_readiness=75,
        )
        assert result.record_id == "rec-gr-1"

    @pytest.mark.asyncio
    async def test_model_drift_event_async(self, async_client):
        async_client._http.post = AsyncMock(
            return_value=_mock_grc_resp("rec-mde-1", "model_drift_event", ["EUAIA_ART72"])
        )
        result = await async_client.model_drift_event(
            "fraud-v2", "f1_score", 0.92, 0.78, 0.85,
            detected_by="monitoring-bot",
        )
        assert result.record_id == "rec-mde-1"

    @pytest.mark.asyncio
    async def test_on_error_raise(self, async_client):
        from mima_governance import MimaAttestationError
        async_client.on_error = "raise"
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        async_client._http.post = AsyncMock(return_value=mock_resp)
        with pytest.raises(MimaAttestationError):
            await async_client.access_review(
                "u", "r", True, reviewed_by="reviewer"
            )

    @pytest.mark.asyncio
    async def test_on_error_warn_returns_empty_record_id(self, async_client):
        async_client.on_error = "warn"
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "unavailable"
        async_client._http.post = AsyncMock(return_value=mock_resp)
        result = await async_client.access_review(
            "u", "r", True, reviewed_by="reviewer"
        )
        assert result.record_id == ""


# ---------------------------------------------------------------------------
# Async attest decorator
# ---------------------------------------------------------------------------

class TestAsyncAttestDecorator:
    @pytest.mark.asyncio
    async def test_attest_wraps_async_fn(self, async_client):
        async_client._http.post = AsyncMock(return_value=_mock_attest_resp())

        @async_client.attest(tool_name="classify")
        async def classify(doc):
            return f"classified:{doc}"

        result = await classify("hello")
        assert result == "classified:hello"
        async_client._http.post.assert_called_once()

    def test_attest_raises_type_error_for_sync_fn(self, async_client):
        with pytest.raises(TypeError, match="async function"):
            @async_client.attest(tool_name="sync_fn")
            def sync_fn(x):
                return x

    @pytest.mark.asyncio
    async def test_attest_batch_mode_enqueues(self, async_client):
        async_client._http.post = AsyncMock(return_value=_mock_attest_resp())

        @async_client.attest(tool_name="classify", mode="batch")
        async def classify(doc):
            return doc

        await classify("hello")
        assert len(async_client._batch_queue) == 1

    @pytest.mark.asyncio
    async def test_attest_result_passed_through(self, async_client):
        async_client._http.post = AsyncMock(return_value=_mock_attest_resp())

        @async_client.attest(tool_name="fn")
        async def fn():
            return {"answer": 42}

        result = await fn()
        assert result == {"answer": 42}


# ---------------------------------------------------------------------------
# Async trace context manager
# ---------------------------------------------------------------------------

class TestAsyncTrace:
    @pytest.mark.asyncio
    async def test_trace_pushes_on_exit(self, async_client):
        async_client._http.post = AsyncMock(return_value=_mock_attest_resp("attest-trace-1"))

        async with async_client.trace("classify_doc") as t:
            t.set_input("hello world")
            t.set_output("positive")

        async_client._http.post.assert_called_once()
        assert t.record_id == "attest-trace-1"

    @pytest.mark.asyncio
    async def test_trace_record_id_populated(self, async_client):
        async_client._http.post = AsyncMock(return_value=_mock_attest_resp("attest-xyz"))

        async with async_client.trace("step") as t:
            t.set_input({"data": 1})

        assert t.record_id == "attest-xyz"

    @pytest.mark.asyncio
    async def test_trace_empty_hashes_on_no_set(self, async_client):
        posted = {}

        async def capture_post(url, *, json, **kwargs):
            posted.update(json)
            return _mock_attest_resp()

        async_client._http.post = capture_post

        async with async_client.trace("empty_trace"):
            pass

        assert posted["input_hash"] == ""
        assert posted["output_hash"] == ""


# ---------------------------------------------------------------------------
# Async batch flush
# ---------------------------------------------------------------------------

class TestAsyncBatchFlush:
    @pytest.mark.asyncio
    async def test_flush_calls_batch_endpoint(self, async_client):
        async_client._http.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            url=MagicMock(__str__=lambda s: "http://localhost:8081/api/workspaces/ws/governance/attestations/batch"),
            json=MagicMock(return_value={"accepted": 3, "rejected": 0, "results": []}),
        ))
        # Manually add 3 records to bypass _enqueue
        from mima_governance.types import AttestationRecord
        for i in range(3):
            async_client._batch_queue.append(AttestationRecord(
                tool_name=f"tool-{i}", input_hash="a" * 64, output_hash="b" * 64
            ))
        await async_client._flush_batch()
        assert async_client._http.post.call_count == 1
        assert async_client._batch_queue == []

    @pytest.mark.asyncio
    async def test_flush_falls_back_on_404(self, async_client):
        call_count = 0

        async def mock_post(url, *, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if "/batch" in str(url):
                resp = MagicMock()
                resp.status_code = 404
                resp.url = MagicMock(__str__=lambda s: url)
                return resp
            return _mock_attest_resp(f"attest-{call_count}")

        async_client._http.post = mock_post

        from mima_governance.types import AttestationRecord
        for i in range(3):
            async_client._batch_queue.append(AttestationRecord(
                tool_name=f"tool-{i}", input_hash="a" * 64, output_hash="b" * 64
            ))

        await async_client._flush_batch()
        # 1 batch attempt + 3 individual fallback calls
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_enqueue_triggers_flush_at_max_size(self, async_client):
        async_client._batch_max_size = 3
        posted_batch = []

        async def mock_post(url, *, json=None, **kwargs):
            if "/batch" in str(url):
                posted_batch.append(json)
                resp = MagicMock()
                resp.status_code = 200
                resp.url = MagicMock(__str__=lambda s: url)
                resp.json.return_value = {"accepted": 3, "rejected": 0, "results": []}
                return resp
            return _mock_attest_resp()

        async_client._http.post = mock_post

        from mima_governance.types import AttestationRecord
        for i in range(3):
            await async_client._enqueue(AttestationRecord(
                tool_name=f"tool-{i}", input_hash="a" * 64, output_hash="b" * 64
            ))

        assert len(posted_batch) == 1
        assert len(posted_batch[0]["records"]) == 3
        assert async_client._batch_queue == []

    @pytest.mark.asyncio
    async def test_context_manager_flushes_on_exit(self, async_client):
        posted_batch = []

        async def mock_post(url, *, json=None, **kwargs):
            if "/batch" in str(url):
                posted_batch.append(json)
                resp = MagicMock()
                resp.status_code = 200
                resp.url = MagicMock(__str__=lambda s: url)
                resp.json.return_value = {"accepted": 2, "rejected": 0, "results": []}
                return resp
            return _mock_attest_resp()

        from mima_governance import AsyncMimaGovernance
        with patch("httpx.AsyncClient") as MockClient:
            http_instance = MagicMock()
            http_instance.post = mock_post
            http_instance.aclose = AsyncMock()
            MockClient.return_value = http_instance

            async with AsyncMimaGovernance(
                workspace_id="ws", api_key="key", system_name="sys",
                base_url="http://localhost:8081",
            ) as mima:
                from mima_governance.types import AttestationRecord
                mima._batch_queue.extend([
                    AttestationRecord(tool_name="t1", input_hash="a" * 64, output_hash="b" * 64),
                    AttestationRecord(tool_name="t2", input_hash="c" * 64, output_hash="d" * 64),
                ])

        assert len(posted_batch) == 1
        assert posted_batch[0]["records"][0]["tool_name"] == "t1"


# ---------------------------------------------------------------------------
# Validation (same rules as sync, exercised via async path)
# ---------------------------------------------------------------------------

class TestAsyncValidation:
    @pytest.mark.asyncio
    async def test_invalid_risk_tier_raises(self, async_client):
        with pytest.raises(ValueError, match="risk_tier"):
            await async_client.ai_risk_assessment(
                "sys", "extreme", "uc",
                impact_domains=["x"],
                art5_self_assessment=True,
                assessor="a",
            )

    @pytest.mark.asyncio
    async def test_invalid_drift_type_raises(self, async_client):
        with pytest.raises(ValueError, match="drift_type"):
            await async_client.model_drift_event(
                "m", "f1", 0.9, 0.7, 0.8,
                drift_type="quantum",
                detected_by="monitor",
            )

    @pytest.mark.asyncio
    async def test_governance_review_readiness_out_of_range(self, async_client):
        with pytest.raises(ValueError, match="readiness"):
            await async_client.governance_review(
                "board", "q1",
                frameworks_reviewed=["soc2"],
                overall_readiness=101,
            )
