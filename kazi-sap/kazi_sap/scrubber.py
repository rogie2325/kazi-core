"""
SAP data scrubbing pipeline.

Three-phase workflow:

  1. analyze()    — fetch records from SAP via OData, run LLM quality
                    analysis, return structured ScrubResult list.
                    No SAP writes. Safe to run against production.

  2. report_text() — format analysis results as a human-readable summary.

  3. remediate()  — apply approved fixes back to SAP via BAPI.
                    Every write is gated behind an approval_callback so
                    nothing commits without human sign-off.
                    High-severity findings are always skipped — they require
                    a separate manual review process.

Usage::

    from kazi import Kazi, KaziConfig
    from kazi_sap import SAPScrubPipeline, ScrubConfig
    from kazi_sap.auth import APIKeyAuth
    from kazi_sap.tools import sap_odata_tool, sap_bapi_tool

    auth = APIKeyAuth(api_key="...")
    async with await Kazi.create(config) as kazi:
        kazi.add_tool(sap_odata_tool(base_url, auth))
        kazi.add_tool(sap_bapi_tool(conn_params, allowed_bapis=["BAPI_VENDOR_CHANGE"]))

        pipeline = SAPScrubPipeline(kazi=kazi, auth=auth, base_url=base_url)

        results = await pipeline.analyze("VendorSet", filter="CountryCode eq 'UK'")
        print(pipeline.report_text(results))

        async def approve(tool_calls):
            print(tool_calls)
            return tool_calls if input("Approve? [y/n]: ") == "y" else None

        report = await pipeline.remediate(results, approval_callback=approve)
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from kazi_sap.models import RemediationReport, ScrubResult
from kazi_sap.prompts import get_prompt

logger = logging.getLogger(__name__)

_MAX_FETCH = 500  # safety cap — prevents accidentally pulling huge tables


@dataclass
class ScrubConfig:
    """
    Configuration for the SAP data scrubbing pipeline.

    concurrency         Parallel LLM calls during the analysis phase.
                        5 is a safe default; increase for larger batches
                        if your LLM rate limit allows.
    auto_fix_severity   Maximum severity level that can be auto-applied
                        without extra human input.
                        "low" is the safest default.  Never set to "high" —
                        high-severity findings always require manual review.
    page_size           Records per OData page during the fetch phase.
    max_records         Hard cap on total records per pipeline run.
    system_prompt       Override the default entity-specific prompt.
                        Leave None to use the built-in prompts from
                        kazi_sap.prompts.
    """

    concurrency: int = 5
    auto_fix_severity: Literal["low", "medium"] = "low"
    page_size: int = 50
    max_records: int = _MAX_FETCH
    system_prompt: str | None = None


class SAPScrubPipeline:
    """
    Read SAP master data → LLM analysis → structured report → optional write-back.
    """

    def __init__(
        self,
        kazi,
        auth,
        base_url: str,
        config: ScrubConfig | None = None,
    ) -> None:
        self._kazi = kazi
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self.config = config or ScrubConfig()

    # ── Public API ────────────────────────────────────────────────────────

    async def analyze(
        self,
        entity_set: str,
        filter: str = "",
        select: str = "",
    ) -> list[ScrubResult]:
        """
        Fetch records from SAP and run LLM data quality analysis.

        Returns a list of ScrubResult — one per record.
        Makes no writes to SAP; safe to run against production.
        """
        records = await self._fetch(entity_set, filter=filter, select=select)
        if not records:
            logger.info("No records returned for %s (filter=%r)", entity_set, filter)
            return []

        logger.info("Analysing %d %s record(s)...", len(records), entity_set)
        prompt = self.config.system_prompt or get_prompt(entity_set)

        messages = [
            f"Analyse this SAP {entity_set} record for data quality issues:\n"
            f"{json.dumps(record, ensure_ascii=False)}"
            for record in records
        ]

        raw = await self._kazi.batch_run(
            messages,
            concurrency=self.config.concurrency,
            system_prompt=prompt,
            response_schema=ScrubResult,
            thread_id_prefix=f"scrub:{entity_set}",
            on_error="skip",
        )

        results: list[ScrubResult] = []
        for i, item in enumerate(raw):
            if isinstance(item, Exception):
                logger.warning("Analysis failed for record %d: %s", i, item)
                continue
            # response_schema returns the parsed model directly or wrapped in RunResult
            if isinstance(item, ScrubResult):
                results.append(item)
            elif hasattr(item, "reply") and isinstance(item.reply, ScrubResult):
                results.append(item.reply)

        clean = sum(1 for r in results if not r.issues)
        logger.info(
            "Analysis complete — %d clean, %d with issues (out of %d analysed)",
            clean, len(results) - clean, len(results),
        )
        return results

    async def remediate(
        self,
        results: list[ScrubResult],
        approval_callback: Callable,
        bapi_name: str = "BAPI_VENDOR_CHANGE",
    ) -> RemediationReport:
        """
        Apply fixes from a previous analyze() call back to SAP.

        Rules:
          - Records with no issues → skipped (clean)
          - High severity → always skipped, logged for manual review
          - auto_fixable=False → pending (needs manual intervention)
          - Severity above auto_fix_severity threshold → pending
          - All other fixable records → gated on approval_callback

        Nothing is written to SAP without the approval_callback returning
        a non-None value.  The callback receives the proposed tool calls
        and must return them (approved) or None (denied).
        """
        _sev_rank = {"low": 0, "medium": 1, "high": 2}
        max_rank = _sev_rank[self.config.auto_fix_severity]

        auto_fixed = pending = skipped_high = errors = 0

        for result in results:
            if not result.issues:
                continue

            if result.severity == "high":
                skipped_high += 1
                logger.warning(
                    "Skipping %s (%s — high severity, manual review required): %s",
                    result.record_id, result.entity_type,
                    [i.issue for i in result.issues],
                )
                continue

            if not result.auto_fixable or _sev_rank[result.severity] > max_rank:
                pending += 1
                continue

            # Build the list of concrete field changes
            fixable = [i for i in result.issues if i.suggested is not None]
            if not fixable:
                pending += 1
                continue

            fix_lines = "\n".join(
                f"  {i.field}: {i.current!r} → {i.suggested!r}"
                for i in fixable
            )
            try:
                await self._kazi.run_with_approval(
                    f"Apply these data quality fixes to SAP {result.entity_type} "
                    f"record {result.record_id}:\n{fix_lines}\n\n"
                    f"Call {bapi_name} with testrun=false to commit the changes.",
                    thread_id=f"scrub:fix:{result.record_id}",
                    approval_callback=approval_callback,
                )
                auto_fixed += 1
                logger.info("Fixed %s %s", result.entity_type, result.record_id)
            except Exception as exc:
                logger.error(
                    "Remediation failed for %s %s: %s",
                    result.entity_type, result.record_id, exc,
                )
                errors += 1

        clean_count = sum(1 for r in results if not r.issues)
        return RemediationReport(
            entity_set="",
            total_analyzed=len(results),
            clean=clean_count,
            with_issues=len(results) - clean_count,
            auto_fixed=auto_fixed,
            pending_approval=pending,
            skipped_high_severity=skipped_high,
            errors=errors,
            results=results,
        )

    def report_text(self, results: list[ScrubResult]) -> str:
        """Format analysis results as a human-readable plain-text report."""
        total = len(results)
        by_sev: dict[str, list[ScrubResult]] = {
            "high": [], "medium": [], "low": [], "clean": []
        }
        for r in results:
            bucket = "clean" if not r.issues else r.severity
            by_sev[bucket].append(r)

        lines = [
            f"SAP Data Quality Report — {total} record(s) analysed",
            "=" * 60,
        ]

        for sev in ("high", "medium", "low"):
            group = by_sev[sev]
            if not group:
                continue
            lines.append(f"\n[{sev.upper()}] — {len(group)} record(s)")
            for r in group:
                fixable = "auto-fixable" if r.auto_fixable else "manual review"
                lines.append(f"  {r.record_id} ({r.entity_type}) [{fixable}]")
                for issue in r.issues:
                    fix = f" → {issue.suggested!r}" if issue.suggested else " (no suggestion)"
                    lines.append(f"    • {issue.field}: {issue.issue}{fix}")
                if r.notes:
                    lines.append(f"    note: {r.notes}")

        lines.append(f"\n✓ Clean records: {len(by_sev['clean'])}")
        lines.append(
            f"  Total issues: {sum(len(r.issues) for r in results)}  |  "
            f"High: {len(by_sev['high'])}  "
            f"Medium: {len(by_sev['medium'])}  "
            f"Low: {len(by_sev['low'])}"
        )
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────

    async def _fetch(
        self,
        entity_set: str,
        filter: str = "",
        select: str = "",
    ) -> list[dict[str, Any]]:
        """Paginate through OData and return up to max_records records."""
        import httpx

        all_records: list[dict] = []
        skip = 0

        while len(all_records) < self.config.max_records:
            params: dict[str, Any] = {
                "$format": "json",
                "$top": self.config.page_size,
                "$skip": skip,
            }
            if filter:
                params["$filter"] = filter
            if select:
                params["$select"] = select

            try:
                auth_headers = await self._auth.headers()
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(
                        f"{self._base_url}/{entity_set}",
                        params=params,
                        headers={**auth_headers, "Accept": "application/json"},
                    )
                    resp.raise_for_status()
            except Exception as exc:
                logger.error("OData fetch failed for %s (skip=%d): %s", entity_set, skip, exc)
                break

            data = resp.json()
            page = data.get("d", {}).get("results", data.get("value", []))
            if not page:
                break

            all_records.extend(page)
            if len(page) < self.config.page_size:
                break  # reached the last page

            skip += len(page)

        capped = all_records[: self.config.max_records]
        if len(all_records) > self.config.max_records:
            logger.warning(
                "Fetch capped at %d records for %s (max_records limit)",
                self.config.max_records, entity_set,
            )
        return capped
