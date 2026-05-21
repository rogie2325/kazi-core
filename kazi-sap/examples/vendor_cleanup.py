"""
End-to-end vendor data cleanup example.

Demonstrates the full three-phase workflow:
  1. analyze()    — fetch records, run LLM quality check (read-only)
  2. report_text() — print a human-readable summary
  3. remediate()  — apply fixes with an interactive CLI approval gate

Usage::

    export SAP_BASE_URL="https://sandbox.api.sap.com/.../API_BUSINESS_PARTNER"
    export SAP_API_KEY="your-api-key"
    export ANTHROPIC_API_KEY="sk-ant-..."

    python examples/vendor_cleanup.py
    python examples/vendor_cleanup.py --filter "CountryCode eq 'DE'" --max-records 100
    python examples/vendor_cleanup.py --dry-run   # analyze only, no remediation
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAP vendor data quality cleanup via kazi-sap")
    p.add_argument("--entity-set", default="VendorSet", help="OData entity set to clean")
    p.add_argument("--filter", default="", help="OData $filter expression")
    p.add_argument("--max-records", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="Analyze only; skip remediation")
    p.add_argument(
        "--auto-fix-severity",
        choices=["low", "medium"],
        default="low",
        help="Maximum severity level for automatic fixes",
    )
    return p.parse_args()


async def _approval_callback(tool_calls: list) -> list | None:
    """Interactive CLI approval gate — shown before each SAP write."""
    print("\n── Proposed SAP changes ──────────────────────────────────────")
    for tc in tool_calls:
        print(f"  {tc}")
    print("──────────────────────────────────────────────────────────────")
    answer = input("Approve these changes? [y/N]: ").strip().lower()
    return tool_calls if answer == "y" else None


async def main() -> None:
    args = _parse_args()

    base_url = os.environ.get("SAP_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("SAP_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not base_url or not api_key or not anthropic_key:
        print(
            "ERROR: SAP_BASE_URL, SAP_API_KEY, and ANTHROPIC_API_KEY must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    from kazi_sap import SAPScrubPipeline, ScrubConfig
    from kazi_sap.auth import APIKeyAuth
    from kazi_sap.tools import sap_odata_tool

    from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider

    auth = APIKeyAuth(api_key=api_key)
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6")
    )

    async with await Kazi.create(config) as kazi:
        kazi.add_tool(
            sap_odata_tool(
                base_url,
                auth,
                allowed_entity_sets=[args.entity_set],
            )
        )

        # BAPI tool is only needed for remediation; conn_params would normally
        # come from environment / secrets manager in a real deployment.
        # Shown here for completeness — left as a no-op placeholder.
        # kazi.add_tool(sap_bapi_tool(conn_params, allowed_bapis=["BAPI_VENDOR_CHANGE"]))

        pipeline = SAPScrubPipeline(
            kazi=kazi,
            auth=auth,
            base_url=base_url,
            config=ScrubConfig(
                max_records=args.max_records,
                auto_fix_severity=args.auto_fix_severity,
            ),
        )

        # ── Phase 1: Analyze ──────────────────────────────────────────────
        print(f"\nAnalysing {args.entity_set}…")
        results = await pipeline.analyze(args.entity_set, filter=args.filter)

        if not results:
            print("No records returned.")
            return

        # ── Phase 2: Report ───────────────────────────────────────────────
        print("\n" + pipeline.report_text(results))

        if args.dry_run:
            print("\n[dry-run] Skipping remediation.")
            return

        # ── Phase 3: Remediate ────────────────────────────────────────────
        issues_found = sum(1 for r in results if r.issues)
        if not issues_found:
            print("\nNo issues to remediate.")
            return

        answer = input(f"\nProceed with remediation of {issues_found} record(s)? [y/N]: ").strip()
        if answer.lower() != "y":
            print("Remediation cancelled.")
            return

        report = await pipeline.remediate(results, approval_callback=_approval_callback)

        print(
            f"\nRemediation complete — "
            f"fixed: {report.auto_fixed}  "
            f"pending: {report.pending_approval}  "
            f"skipped (high): {report.skipped_high_severity}  "
            f"errors: {report.errors}"
        )


if __name__ == "__main__":
    asyncio.run(main())
