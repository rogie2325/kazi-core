"""
Shared fixtures for kazi-sap tests.

Three fixture groups:
  mock_sap     — in-memory OData server via respx; no real SAP required
  mock_kazi   — minimal Kazi stand-in with controllable batch_run / run_with_approval
  sample_*     — pre-built ScrubResult lists for unit testing remediate() and report_text()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from kazi_sap.auth import APIKeyAuth
from kazi_sap.models import ScrubIssue, ScrubResult
from kazi_sap.scrubber import SAPScrubPipeline, ScrubConfig

# ── Paths ─────────────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


# ── SAP auth fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def sap_auth() -> APIKeyAuth:
    return APIKeyAuth(api_key="test-key-12345")


@pytest.fixture
def base_url() -> str:
    return "https://sandbox.api.sap.com/test/odata/v2/API_BUSINESS_PARTNER"


# ── Dirty record fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def dirty_vendors() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "dirty_vendors.json").read_text())


@pytest.fixture
def dirty_customers() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "dirty_customers.json").read_text())


# ── Pre-built ScrubResult samples ─────────────────────────────────────────────

@pytest.fixture
def sample_results() -> list[ScrubResult]:
    """Four results: clean, low-auto, medium-manual, high-skip."""
    return [
        ScrubResult(
            record_id="V000",
            entity_type="VendorSet",
            issues=[],
            severity="low",
            auto_fixable=True,
            notes="All fields valid.",
        ),
        ScrubResult(
            record_id="V001",
            entity_type="VendorSet",
            issues=[
                ScrubIssue(
                    field="CountryCode",
                    issue="'UK' is not a valid ISO 3166-1 alpha-2 code; should be 'GB'.",
                    current="UK",
                    suggested="GB",
                )
            ],
            severity="low",
            auto_fixable=True,
            notes="",
        ),
        ScrubResult(
            record_id="V002",
            entity_type="VendorSet",
            issues=[
                ScrubIssue(
                    field="TaxNumber",
                    issue="Tax number is blank for a DE vendor; VAT registration is required.",
                    current="",
                    suggested=None,
                )
            ],
            severity="medium",
            auto_fixable=False,
            notes="Needs manual tax team input.",
        ),
        ScrubResult(
            record_id="V004",
            entity_type="VendorSet",
            issues=[
                ScrubIssue(
                    field="IBAN",
                    issue="IBAN format is invalid and does not pass checksum validation.",
                    current="INVALID-IBAN-VALUE",
                    suggested=None,
                ),
                ScrubIssue(
                    field="Email",
                    issue="Email domain 'fastship' has no TLD — likely incomplete.",
                    current="billing@fastship",
                    suggested=None,
                ),
            ],
            severity="high",
            auto_fixable=False,
            notes="Multiple critical fields require manual correction.",
        ),
    ]


# ── Mock Kazi ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_kazi(sample_results):
    """
    Minimal Kazi double.

    batch_run returns the sample_results list (minus the clean record,
    simulating that the real pipeline may return fewer items on error).
    run_with_approval calls the approval_callback and commits if approved.
    """
    kazi = MagicMock()

    # batch_run returns ScrubResult objects directly (simulates response_schema path)
    async def _batch_run(messages, **kwargs):
        return list(sample_results)

    kazi.batch_run = AsyncMock(side_effect=_batch_run)

    async def _run_with_approval(prompt, *, thread_id, approval_callback):
        tool_calls = [{"prompt": prompt}]
        approved = await approval_callback(tool_calls)
        if approved is None:
            raise RuntimeError("Approval denied")

    kazi.run_with_approval = AsyncMock(side_effect=_run_with_approval)

    return kazi


# ── Pipeline factory ───────────────────────────────────────────────────────────

@pytest.fixture
def pipeline(mock_kazi, sap_auth, base_url):
    return SAPScrubPipeline(
        kazi=mock_kazi,
        auth=sap_auth,
        base_url=base_url,
        config=ScrubConfig(concurrency=2, auto_fix_severity="low"),
    )
