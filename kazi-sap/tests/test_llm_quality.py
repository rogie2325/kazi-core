"""
LLM quality tests — run against a real LLM with mocked SAP transport.

These tests verify that the prompts actually produce sensible structured
output, not just that the plumbing works.  They cost real tokens so they
are marked @pytest.mark.llm and excluded from the default pytest run.

Run with:
    pytest -m llm tests/test_llm_quality.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from kazi_sap.scrubber import SAPScrubPipeline, ScrubConfig

pytestmark = pytest.mark.llm

FIXTURES = Path(__file__).parent / "fixtures"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"Required env vars not set: {', '.join(missing)}")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def real_kazi():
    """
    Real Kazi instance backed by Anthropic Claude.

    Requires ANTHROPIC_API_KEY in environment.
    The Kazi import is deferred so the test file can be collected
    even when the kazi package is not installed in CI.
    """
    _require_env("ANTHROPIC_API_KEY")
    from kazi import KaziConfig, LLMConfig, LLMProvider

    config = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # cheap; good enough for quality tests
        )
    )
    # We create the Kazi synchronously here; the fixture is sync but async
    # creation is handled inside the test via asyncio.
    return config


@pytest.fixture
def dirty_vendors():
    return json.loads((FIXTURES / "dirty_vendors.json").read_text())


@pytest.fixture
def dirty_customers():
    return json.loads((FIXTURES / "dirty_customers.json").read_text())


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_uk_country_code_flagged(real_kazi, dirty_vendors):
    """
    'UK' is not a valid ISO 3166-1 code; the LLM should flag it and suggest 'GB'.
    """
    from kazi_sap.auth import APIKeyAuth

    from kazi import Kazi

    auth = APIKeyAuth(api_key="dummy")
    async with await Kazi.create(real_kazi) as kazi:
        pipeline = SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url="https://example.com",
            config=ScrubConfig(concurrency=1),
        )
        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=[dirty_vendors[0]])):
            results = await pipeline.analyze("VendorSet")

    assert results, "Expected at least one ScrubResult"
    r = results[0]
    assert r.issues, "Expected CountryCode='UK' to produce at least one issue"
    fields = [i.field.lower() for i in r.issues]
    assert any("country" in f for f in fields), f"No CountryCode issue found; got fields: {fields}"
    # Suggested fix should be GB
    country_issues = [i for i in r.issues if "country" in i.field.lower()]
    assert any(i.suggested == "GB" for i in country_issues), (
        f"Expected suggested='GB' but got: {[i.suggested for i in country_issues]}"
    )


@pytest.mark.asyncio
async def test_duplicate_vendors_flagged(real_kazi, dirty_vendors):
    """V002 and V003 are identical — the LLM should flag both as potential duplicates."""
    from kazi_sap.auth import APIKeyAuth

    from kazi import Kazi

    auth = APIKeyAuth(api_key="dummy")
    duplicates = dirty_vendors[1:3]  # V002 and V003

    async with await Kazi.create(real_kazi) as kazi:
        pipeline = SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url="https://example.com",
            config=ScrubConfig(concurrency=2),
        )
        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=duplicates)):
            results = await pipeline.analyze("VendorSet")

    assert len(results) == 2
    issue_texts = " ".join(
        i.issue.lower() for r in results for i in r.issues
    )
    assert "duplicate" in issue_texts or "identical" in issue_texts, (
        f"Expected duplicate warning; got issue texts: {issue_texts!r}"
    )


@pytest.mark.asyncio
async def test_invalid_iban_flagged(real_kazi, dirty_vendors):
    """V004 has 'INVALID-IBAN-VALUE' — the LLM should flag it as an invalid IBAN."""
    from kazi_sap.auth import APIKeyAuth

    from kazi import Kazi

    auth = APIKeyAuth(api_key="dummy")
    async with await Kazi.create(real_kazi) as kazi:
        pipeline = SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url="https://example.com",
            config=ScrubConfig(concurrency=1),
        )
        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=[dirty_vendors[3]])):
            results = await pipeline.analyze("VendorSet")

    assert results
    issue_texts = " ".join(i.issue.lower() for r in results for i in r.issues)
    assert "iban" in issue_texts, f"Expected IBAN issue; got: {issue_texts!r}"


@pytest.mark.asyncio
async def test_clean_vendor_no_issues(real_kazi):
    """A well-formed vendor record should come back with no issues."""
    from kazi_sap.auth import APIKeyAuth

    from kazi import Kazi

    clean_vendor = {
        "VendorId": "V999",
        "Name": "Perfect Vendor GmbH",
        "CountryCode": "DE",
        "TaxNumber": "DE123456789",
        "PaymentTerms": "NET30",
        "IBAN": "DE89370400440532013000",
        "Email": "billing@perfect-vendor.de",
        "Phone": "+49 89 12345678",
        "City": "Munich",
        "PostalCode": "80331",
    }
    auth = APIKeyAuth(api_key="dummy")
    async with await Kazi.create(real_kazi) as kazi:
        pipeline = SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url="https://example.com",
            config=ScrubConfig(concurrency=1),
        )
        with patch.object(pipeline, "_fetch", new=AsyncMock(return_value=[clean_vendor])):
            results = await pipeline.analyze("VendorSet")

    assert results
    r = results[0]
    assert not r.issues or r.severity == "low", (
        f"Expected clean or low-severity result for a well-formed vendor; "
        f"got severity={r.severity!r}, issues={r.issues}"
    )
