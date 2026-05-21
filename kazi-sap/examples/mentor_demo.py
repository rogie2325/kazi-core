"""
kazi-sap  —  Mentor Demo
============================================================
Three-phase SAP master data quality pipeline:

  Phase 1  analyze()    — read vendor records, run LLM quality check
  Phase 2  report_text() — structured findings by severity
  Phase 3  remediate()  — field-level fixes with human approval gate

Run modes
---------
  --mock       Offline demo with realistic pre-baked SAP data (default)
  --live       Connect to a real SAP OData endpoint (requires env vars)

Live mode env vars:
  SAP_BASE_URL       https://sandbox.api.sap.com/.../API_BUSINESS_PARTNER
  SAP_API_KEY        your-api-key
  ANTHROPIC_API_KEY  sk-ant-...

Usage::

    python examples/mentor_demo.py             # mock mode
    python examples/mentor_demo.py --mock      # explicit mock
    python examples/mentor_demo.py --live      # real SAP
    python examples/mentor_demo.py --live --dry-run   # real SAP, no writes
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s — %(message)s")

# ── Terminal helpers ──────────────────────────────────────────────────────────

W = 64  # box width


def _banner() -> None:
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  kazi-sap  ·  Vendor Data Quality Demo".center(W) + "║")
    print("╚" + "═" * W + "╝")
    print()


def _section(title: str) -> None:
    print()
    print("┌─  " + title + "  " + "─" * max(0, W - len(title) - 5) + "┐")


def _rule() -> None:
    print("└" + "─" * W + "┘")


def _tick(msg: str) -> None:
    print(f"  ✓  {msg}")


# ── Mock SAP data ─────────────────────────────────────────────────────────────
# Realistic vendor records that cover every issue category the LLM checks for.
# A real SAP expert will recognise these patterns immediately.

MOCK_VENDORS = [
    {
        "VendorId": "0001000123",
        "Name": "Müller Hydraulik GmbH ",       # trailing whitespace
        "CountryCode": "GER",                   # wrong: not ISO 3166-1 alpha-2
        "Street": "Industriestraße 42",
        "PostalCode": "80331",
        "City": "München",
        "PaymentTerms": "0001",
        "PaymentMethod": "T",
        "BankCountry": "DE",
        "IBAN": "DE89370400440532013000",
        "BIC": "COBADEFFXXX",
        "ReconciliationAccount": "160000",
    },
    {
        "VendorId": "0001000124",
        "Name": "Miller Hydraulics GmbH",        # probable duplicate of 0001000123
        "CountryCode": "DE",
        "Street": "Industriestrasse 42",          # same address, different encoding
        "PostalCode": "80331",
        "City": "Munich",
        "PaymentTerms": "TBD",                   # placeholder — not a valid SAP key
        "PaymentMethod": "T",
        "BankCountry": "DE",
        "IBAN": "",                              # missing — payment method T requires it
        "BIC": "",
        "ReconciliationAccount": "160000",
    },
    {
        "VendorId": "0001000125",
        "Name": "Acme Supplies Ltd",
        "CountryCode": "UK",                     # wrong: should be GB
        "Street": "14 Regent Street",
        "PostalCode": "SW1Y 4PH",
        "City": "London",
        "PaymentTerms": "NT30",
        "PaymentMethod": "C",
        "BankCountry": "GB",
        "IBAN": "GB29NWBK60161331926819",
        "BIC": "NWBKGB2L",
        "ReconciliationAccount": "161000",
    },
    {
        "VendorId": "0001000126",
        "Name": "  TechParts AG",                # leading whitespace
        "CountryCode": "CH",
        "Street": "Bahnhofstrasse 12",
        "PostalCode": "8001",
        "City": "Zürich",
        "PaymentTerms": "ZB14",
        "PaymentMethod": "T",
        "BankCountry": "CH",
        "IBAN": "CH9300762011623852957",
        "BIC": "UBSWCHZH80A",
        "ReconciliationAccount": "155000",       # outside 160000–169999 standard range
    },
    {
        "VendorId": "0001000127",
        "Name": "Global Materials BV",
        "CountryCode": "NL",
        "Street": "Keizersgracht 126",
        "PostalCode": "1015 CW",
        "City": "Amsterdam",
        "PaymentTerms": "0030",
        "PaymentMethod": "T",
        "BankCountry": "NL",
        "IBAN": "NL91ABNA0417164300",
        "BIC": "ABNANL2A",
        "ReconciliationAccount": "162000",
    },
]

# Pre-baked LLM analysis — exactly what Claude returns for the records above.

def _mock_results():
    from kazi_sap.models import ScrubIssue, ScrubResult
    return [
        ScrubResult(
            record_id="0001000123",
            entity_type="Vendor",
            issues=[
                ScrubIssue(
                    field="CountryCode",
                    current="GER",
                    issue="'GER' is not a valid ISO 3166-1 alpha-2 code. Germany uses 'DE'.",
                    suggested="DE",
                ),
                ScrubIssue(
                    field="Name",
                    current="Müller Hydraulik GmbH ",
                    issue="Trailing whitespace in vendor name.",
                    suggested="Müller Hydraulik GmbH",
                ),
            ],
            severity="medium",
            auto_fixable=True,
            notes="CountryCode mismatch may affect payment routing rules.",
        ),
        ScrubResult(
            record_id="0001000124",
            entity_type="Vendor",
            issues=[
                ScrubIssue(
                    field="Name",
                    current="Miller Hydraulics GmbH",
                    issue=(
                        "Possible duplicate of 0001000123 (Müller Hydraulik GmbH) — "
                        "identical address, near-identical name."
                    ),
                    suggested=None,
                ),
                ScrubIssue(
                    field="PaymentTerms",
                    current="TBD",
                    issue="Placeholder value. Must be a valid SAP payment terms key.",
                    suggested=None,
                ),
                ScrubIssue(
                    field="IBAN",
                    current="",
                    issue=(
                        "IBAN is missing. Vendor has PaymentMethod='T' (bank transfer) "
                        "but no bank details on record."
                    ),
                    suggested=None,
                ),
            ],
            severity="high",
            auto_fixable=False,
            notes=(
                "Suspected duplicate vendor. Block payments until the master data team "
                "confirms this is not a duplicate entry."
            ),
        ),
        ScrubResult(
            record_id="0001000125",
            entity_type="Vendor",
            issues=[
                ScrubIssue(
                    field="CountryCode",
                    current="UK",
                    issue="'UK' is not a valid ISO 3166-1 alpha-2 code. Great Britain uses 'GB'.",
                    suggested="GB",
                ),
            ],
            severity="medium",
            auto_fixable=True,
            notes=None,
        ),
        ScrubResult(
            record_id="0001000126",
            entity_type="Vendor",
            issues=[
                ScrubIssue(
                    field="Name",
                    current="  TechParts AG",
                    issue="Leading whitespace in vendor name.",
                    suggested="TechParts AG",
                ),
                ScrubIssue(
                    field="ReconciliationAccount",
                    current="155000",
                    issue=(
                        "AP reconciliation account 155000 is outside the standard "
                        "SAP range 160000–169999. Verify with the GL team."
                    ),
                    suggested=None,
                ),
            ],
            severity="medium",
            auto_fixable=False,
            notes="ReconciliationAccount must be confirmed by GL before correction.",
        ),
        ScrubResult(
            record_id="0001000127",
            entity_type="Vendor",
            issues=[],
            severity="low",
            auto_fixable=True,
            notes=None,
        ),
    ]


# ── Approval callback ─────────────────────────────────────────────────────────

async def _approval_callback(tool_calls: list) -> list | None:
    print()
    print("  ┌─  Proposed SAP write  " + "─" * 39 + "┐")
    for tc in tool_calls:
        print(f"  │  {tc}")
    print("  └" + "─" * 62 + "┘")
    answer = input("  Approve this change? [y/N]: ").strip().lower()
    return tool_calls if answer == "y" else None


# ── Live pipeline ─────────────────────────────────────────────────────────────

async def _run_live(args: argparse.Namespace) -> None:
    base_url   = os.environ.get("SAP_BASE_URL", "").rstrip("/")
    api_key    = os.environ.get("SAP_API_KEY", "")
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")

    missing = [k for k, v in [
        ("SAP_BASE_URL", base_url),
        ("SAP_API_KEY", api_key),
        ("ANTHROPIC_API_KEY", claude_key),
    ] if not v]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
        print("Run with --mock for an offline demo.", file=sys.stderr)
        sys.exit(1)

    from kazi_sap import SAPScrubPipeline, ScrubConfig
    from kazi_sap.auth import APIKeyAuth
    from kazi_sap.tools import sap_odata_tool

    from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider

    auth   = APIKeyAuth(api_key=api_key)
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6")
    )

    async with await Kazi.create(config) as kazi:
        kazi.add_tool(sap_odata_tool(base_url, auth, allowed_entity_sets=["VendorSet"]))

        pipeline = SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url=base_url,
            config=ScrubConfig(max_records=args.max_records, auto_fix_severity="low"),
        )

        _section("Phase 1 — Fetching & Analysing VendorSet")
        t0 = time.monotonic()
        results = await pipeline.analyze("VendorSet", filter=args.filter)
        elapsed = time.monotonic() - t0
        _tick(f"{len(results)} record(s) analysed in {elapsed:.1f}s")
        _rule()

        _section("Phase 2 — Data Quality Report")
        print()
        print(pipeline.report_text(results))
        _rule()

        if args.dry_run:
            print("\n  [dry-run mode] Skipping remediation.")
            return

        await _remediate(pipeline, results)


# ── Mock pipeline ─────────────────────────────────────────────────────────────

async def _run_mock(args: argparse.Namespace) -> None:
    from kazi_sap.scrubber import SAPScrubPipeline, ScrubConfig

    # We never call Kazi in mock mode — no credentials needed.
    pipeline = SAPScrubPipeline(
        kazi=None,   # not used in mock path
        auth=None,
        base_url="https://demo.example.sap.com",
        config=ScrubConfig(max_records=50, auto_fix_severity="low"),
    )

    _section(f"Phase 1 — Analysing {len(MOCK_VENDORS)} vendor records  [mock]")
    print()
    print("  Vendor records loaded:")
    for v in MOCK_VENDORS:
        print(f"    {v['VendorId']}  {v['Name'].strip()}")
    print()
    print("  Running LLM data quality analysis…")
    await asyncio.sleep(0.6)   # simulate LLM latency for the demo
    results = _mock_results()
    issues_count = sum(len(r.issues) for r in results)
    _tick(f"{len(results)} records analysed — {issues_count} issues found")
    _rule()

    _section("Phase 2 — Data Quality Report")
    print()
    print(pipeline.report_text(results))
    _rule()

    if args.dry_run:
        print("\n  [dry-run] Skipping remediation phase.")
        return

    await _remediate(pipeline, results)


# ── Shared remediation ────────────────────────────────────────────────────────

async def _remediate(pipeline, results) -> None:

    issues_found = sum(1 for r in results if r.issues)
    if not issues_found:
        print("\n  No issues to remediate — all records are clean.")
        return

    _section(f"Phase 3 — Remediation  ({issues_found} record(s) with issues)")
    print()
    print("  Safety rules:")
    print("    • HIGH severity  → always skipped, flagged for manual review")
    print("    • No suggestion  → queued as pending (cannot auto-fix)")
    print("    • LOW / MEDIUM   → presented for your approval before any SAP write")
    print()

    answer = input("  Proceed with remediation? [y/N]: ").strip().lower()
    if answer != "y":
        print("\n  Remediation cancelled.")
        _rule()
        return

    # In mock mode, pipeline.kazi is None so we simulate approval flow directly.
    if pipeline._kazi is None:
        await _mock_remediate(results)
    else:
        report = await pipeline.remediate(results, approval_callback=_approval_callback)
        _print_remediation_summary(report)

    _rule()


async def _mock_remediate(results) -> None:
    auto_fixed = pending = skipped_high = 0

    for r in results:
        if not r.issues:
            continue
        if r.severity == "high":
            skipped_high += 1
            print(f"  ⚠  {r.record_id} — SKIPPED (high severity, manual review required)")
            for issue in r.issues:
                print(f"       {issue.field}: {issue.issue}")
            continue

        if not r.auto_fixable:
            pending += 1
            print(f"  ·  {r.record_id} — PENDING (ambiguous fix, no suggested value)")
            continue

        fixable = [i for i in r.issues if i.suggested is not None]
        fix_desc = "  |  ".join(
            f"{i.field}: {i.current!r} → {i.suggested!r}" for i in fixable
        )

        print()
        print(f"  ┌─  Fix proposal: {r.record_id}  {'─' * max(0, 44 - len(r.record_id))}┐")
        for i in fixable:
            print(f"  │    {i.field:<30} {i.current!r:>12} → {i.suggested!r}")
        print(f"  └{'─' * 62}┘")

        answer = input("  Approve? [y/N]: ").strip().lower()
        if answer == "y":
            print(f"  ✓  {r.record_id} — fixed")
            auto_fixed += 1
        else:
            pending += 1
            print(f"  ·  {r.record_id} — skipped by user")

    print()
    print("  ─" * 32)
    print(f"  Results:  fixed {auto_fixed}  ·  pending {pending}  ·  high-sev skipped {skipped_high}")


def _print_remediation_summary(report) -> None:
    print()
    print("  ─" * 32)
    print(
        f"  Results:  "
        f"fixed {report.auto_fixed}  ·  "
        f"pending {report.pending_approval}  ·  "
        f"high-sev skipped {report.skipped_high_severity}  ·  "
        f"errors {report.errors}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="kazi-sap mentor demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=True,
                      help="Offline demo using pre-baked SAP data (default)")
    mode.add_argument("--live", action="store_true",
                      help="Connect to a real SAP OData endpoint (requires env vars)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run analysis and report only — skip remediation")
    p.add_argument("--filter", default="",
                   help="OData $filter expression (live mode only)")
    p.add_argument("--max-records", type=int, default=50,
                   help="Max records to fetch (live mode only)")
    args = p.parse_args()
    if args.live:
        args.mock = False
    return args


async def main() -> None:
    args = _parse_args()
    _banner()

    mode_label = "MOCK (offline)" if args.mock else "LIVE (SAP OData)"
    print(f"  Mode    : {mode_label}")
    print(f"  Dry-run : {'yes — no SAP writes' if args.dry_run else 'no  — remediation enabled'}")

    if args.mock:
        await _run_mock(args)
    else:
        await _run_live(args)

    print()
    print("  Demo complete.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
