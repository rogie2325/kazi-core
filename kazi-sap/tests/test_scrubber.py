"""
Unit tests for SAPScrubPipeline.

All SAP I/O and Kazi LLM calls are mocked — no real network required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from kazi_sap.models import ScrubResult

# ── analyze() ─────────────────────────────────────────────────────────────────

class TestAnalyze:
    @pytest.mark.asyncio
    async def test_returns_scrub_results(self, pipeline, dirty_vendors):
        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=dirty_vendors)):
            results = await pipeline.analyze("VendorSet")

        assert len(results) == 4  # mock_kazi returns all 4 sample_results
        assert all(isinstance(r, ScrubResult) for r in results)

    @pytest.mark.asyncio
    async def test_empty_entity_set_returns_empty_list(self, pipeline):
        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=[])):
            results = await pipeline.analyze("VendorSet")

        assert results == []
        pipeline._kazi.batch_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt(self, pipeline, dirty_vendors):
        pipeline.config.system_prompt = "Custom prompt."
        captured: list[str] = []

        async def _capture(messages, *, system_prompt, **kwargs):
            captured.append(system_prompt)
            return []

        pipeline._kazi.batch_run = AsyncMock(side_effect=_capture)

        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=dirty_vendors)):
            await pipeline.analyze("VendorSet")

        assert captured == ["Custom prompt."]

    @pytest.mark.asyncio
    async def test_skips_exception_items(self, pipeline, dirty_vendors):
        """If batch_run returns an Exception for one item, it should be silently skipped."""
        from kazi_sap.models import ScrubIssue, ScrubResult

        good = ScrubResult(
            record_id="V001", entity_type="VendorSet",
            issues=[ScrubIssue(field="CountryCode", issue="UK→GB", current="UK", suggested="GB")],
            severity="low", auto_fixable=True,
        )

        async def _mixed(messages, **kwargs):
            return [good, ValueError("LLM timeout")]

        pipeline._kazi.batch_run = AsyncMock(side_effect=_mixed)

        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=dirty_vendors[:2])):
            results = await pipeline.analyze("VendorSet")

        assert len(results) == 1
        assert results[0].record_id == "V001"


# ── remediate() ───────────────────────────────────────────────────────────────

class TestRemediate:
    @pytest.mark.asyncio
    async def test_clean_records_skipped(self, pipeline, sample_results):
        report = await pipeline.remediate(
            sample_results,
            approval_callback=AsyncMock(return_value=[{}]),
        )
        assert report.clean == 1

    @pytest.mark.asyncio
    async def test_high_severity_always_skipped(self, pipeline, sample_results):
        report = await pipeline.remediate(
            sample_results,
            approval_callback=AsyncMock(return_value=[{}]),
        )
        assert report.skipped_high_severity == 1

    @pytest.mark.asyncio
    async def test_auto_fix_low_severity(self, pipeline, sample_results):
        approved_calls: list = []

        async def approve(tool_calls):
            approved_calls.extend(tool_calls)
            return tool_calls

        report = await pipeline.remediate(sample_results, approval_callback=approve)
        # V001 is low + auto_fixable → should be fixed
        assert report.auto_fixed == 1
        assert len(approved_calls) == 1

    @pytest.mark.asyncio
    async def test_medium_not_auto_fixed_with_low_threshold(self, pipeline, sample_results):
        report = await pipeline.remediate(
            sample_results,
            approval_callback=AsyncMock(return_value=[{}]),
        )
        # V002 is medium + auto_fixable=False → pending
        assert report.pending_approval >= 1

    @pytest.mark.asyncio
    async def test_denied_approval_counts_as_error(self, pipeline, sample_results):
        async def deny(_tool_calls):
            return None  # denied

        report = await pipeline.remediate(sample_results, approval_callback=deny)
        assert report.errors == 1

    @pytest.mark.asyncio
    async def test_report_totals_consistent(self, pipeline, sample_results):
        report = await pipeline.remediate(
            sample_results,
            approval_callback=AsyncMock(return_value=[{}]),
        )
        accounted = (
            report.clean
            + report.auto_fixed
            + report.pending_approval
            + report.skipped_high_severity
            + report.errors
        )
        assert accounted == report.total_analyzed


# ── report_text() ─────────────────────────────────────────────────────────────

class TestReportText:
    def test_includes_all_severities(self, pipeline, sample_results):
        text = pipeline.report_text(sample_results)
        assert "[HIGH]" in text
        assert "[MEDIUM]" in text
        assert "[LOW]" in text

    def test_clean_count_shown(self, pipeline, sample_results):
        text = pipeline.report_text(sample_results)
        assert "Clean records: 1" in text

    def test_empty_results(self, pipeline):
        text = pipeline.report_text([])
        assert "0 record(s) analysed" in text
        assert "Clean records: 0" in text

    def test_auto_fixable_label(self, pipeline, sample_results):
        text = pipeline.report_text(sample_results)
        assert "auto-fixable" in text
        assert "manual review" in text


# ── _fetch() ──────────────────────────────────────────────────────────────────

class TestFetch:
    @pytest.mark.asyncio
    async def test_paginates_until_short_page(self, pipeline):
        """Stops fetching when a page shorter than page_size is returned."""
        page1 = [{"VendorId": str(i)} for i in range(50)]
        page2 = [{"VendorId": str(i)} for i in range(50, 63)]  # only 13 records

        call_count = 0

        import httpx
        import respx

        with respx.mock:
            def _side_effect(request):
                nonlocal call_count
                call_count += 1
                skip = int(dict(
                    kv.split("=") for kv in request.url.query.decode().split("&") if "=" in kv
                ).get("%24skip", "0"))
                page = page2 if skip >= 50 else page1
                return httpx.Response(200, json={"d": {"results": page}})

            respx.get(url__regex=r".*/VendorSet.*").mock(side_effect=_side_effect)
            records = await pipeline._fetch("VendorSet")

        assert len(records) == 63
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_respects_max_records(self, pipeline):
        pipeline.config.max_records = 10
        pipeline.config.page_size = 50

        import httpx
        import respx

        big_page = [{"VendorId": str(i)} for i in range(50)]
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(200, json={"d": {"results": big_page}})
            )
            records = await pipeline._fetch("VendorSet")

        assert len(records) == 10
