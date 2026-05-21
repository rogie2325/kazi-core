"""
Integration tests — run against a real SAP sandbox with TESTRUN=X safety gate.

All BAPI calls use testrun=True (the default), so nothing is committed.

Environment variables required:
    SAP_BASE_URL       OData service root (e.g. https://sandbox.api.sap.com/...)
    SAP_API_KEY        API key for SAP sandbox
    ANTHROPIC_API_KEY  LLM key for the Kazi pipeline

Run with:
    pytest -m integration tests/test_integration.py -v
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"Required env vars not set: {', '.join(missing)}")


@pytest.fixture
def sap_config():
    _require_env("SAP_BASE_URL", "SAP_API_KEY", "ANTHROPIC_API_KEY")
    return {
        "base_url": os.environ["SAP_BASE_URL"],
        "api_key": os.environ["SAP_API_KEY"],
    }


@pytest.fixture
async def live_pipeline(sap_config):
    from kazi_sap.auth import APIKeyAuth
    from kazi_sap.scrubber import SAPScrubPipeline, ScrubConfig
    from kazi_sap.tools import sap_odata_tool

    from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider

    auth = APIKeyAuth(api_key=sap_config["api_key"])
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    )

    async with await Kazi.create(config) as kazi:
        kazi.add_tool(
            sap_odata_tool(
                sap_config["base_url"],
                auth,
                allowed_entity_sets=["A_Supplier", "A_Customer"],
            )
        )
        yield SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url=sap_config["base_url"],
            config=ScrubConfig(concurrency=2, page_size=10, max_records=20),
        )


@pytest.mark.asyncio
async def test_fetch_suppliers_from_sandbox(live_pipeline):
    """Verify we can page records from the SAP sandbox OData service."""
    records = await live_pipeline._fetch("A_Supplier", select="Supplier,SupplierName,Country")
    assert isinstance(records, list), "Expected a list of records"
    # Sandbox may have 0 records — just confirm the call didn't raise
    if records:
        assert "Supplier" in records[0] or "SupplierName" in records[0], (
            f"Unexpected record shape: {records[0]}"
        )


@pytest.mark.asyncio
async def test_analyze_does_not_write(live_pipeline):
    """analyze() must return structured results without committing anything to SAP."""
    results = await live_pipeline.analyze("A_Supplier", select="Supplier,SupplierName,Country")
    # All we require is that the call completed and returned a list
    assert isinstance(results, list)
    # No BAPI tool should have been called
    live_pipeline._kazi.run_with_approval.assert_not_called()


@pytest.mark.asyncio
async def test_odata_filter(live_pipeline):
    """OData $filter is forwarded correctly and the sandbox honours it."""
    records = await live_pipeline._fetch(
        "A_Supplier",
        filter="Country eq 'DE'",
        select="Supplier,Country",
    )
    for rec in records:
        assert rec.get("Country") == "DE", f"Filter not applied: {rec}"
