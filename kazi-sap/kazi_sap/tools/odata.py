"""
SAP OData query tool for kazi.

Registers a single ``sap_odata_query`` tool that the LLM can call to read
any OData entity set from a SAP system.  Supports $filter, $select, $top,
and $skip so the LLM can paginate and filter without additional tooling.

Security defaults
-----------------
- allowed_entity_sets whitelist prevents the LLM from reading entity sets
  outside what the deployment explicitly permits.
- $top is capped at _MAX_TOP regardless of what the LLM requests.
- TLS verification is ON by default.
- Auth credentials never appear in tool metadata or error messages.
"""
from __future__ import annotations

import logging

import httpx

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource
from kazi_sap.auth import SAPAuth

logger = logging.getLogger(__name__)

_MAX_TOP = 500  # hard cap — prevents accidental full-table scans


def sap_odata_tool(
    base_url: str,
    auth: SAPAuth,
    *,
    allowed_entity_sets: list[str] | None = None,
    timeout: int = 30,
    verify_tls: bool = True,
) -> ToolDefinition:
    """
    Return a ToolDefinition for querying SAP data via OData.

    base_url
        OData service root URL, e.g.:
        "https://sandbox.api.sap.com/s4hanacloud/sap/opu/odata/sap/API_BUSINESS_PARTNER"
    auth
        Authentication strategy — APIKeyAuth, XSUAAAuth, or BasicAuth.
    allowed_entity_sets
        Whitelist of entity sets the LLM is allowed to query.
        None = allow all (not recommended in production).
    timeout
        Per-request HTTP timeout in seconds.
    verify_tls
        TLS certificate verification.  Always True in production.
    """
    _base = base_url.rstrip("/")
    _allowed = set(allowed_entity_sets) if allowed_entity_sets else None

    async def _query(
        entity_set: str,
        filter: str = "",
        select: str = "",
        top: int = 20,
        skip: int = 0,
    ) -> str:
        if _allowed is not None and entity_set not in _allowed:
            return (
                f"Entity set '{entity_set}' is not in the allowed list. "
                f"Permitted: {sorted(_allowed)}"
            )

        top = max(1, min(top, _MAX_TOP))
        params: dict[str, str | int] = {
            "$format": "json",
            "$top": top,
            "$skip": skip,
        }
        if filter:
            params["$filter"] = filter
        if select:
            params["$select"] = select

        try:
            auth_headers = await auth.headers()
            async with httpx.AsyncClient(timeout=timeout, verify=verify_tls) as client:
                resp = await client.get(
                    f"{_base}/{entity_set}",
                    params=params,
                    headers={**auth_headers, "Accept": "application/json"},
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return (
                f"SAP OData error HTTP {exc.response.status_code} "
                f"on {entity_set}: {exc.response.text[:300]}"
            )
        except httpx.TimeoutException:
            return f"SAP OData request timed out after {timeout}s for {entity_set}"
        except Exception as exc:
            return f"SAP OData request failed: {exc}"

        data = resp.json()
        # Handle both OData v2 (d.results) and v4 (value) response envelopes
        records = data.get("d", {}).get("results", data.get("value", []))
        if not records:
            return f"No records returned from {entity_set}."

        lines = [f"[{entity_set}] {len(records)} record(s):"]
        for record in records:
            # Strip OData metadata noise (__metadata, __deferred, etc.)
            clean = {k: v for k, v in record.items() if not k.startswith("__")}
            lines.append(str(clean))
        return "\n".join(lines)

    return ToolDefinition(
        name="sap_odata_query",
        description=(
            "Query SAP master data and transactional data via OData. "
            "Entity sets include VendorSet, CustomerSet, MaterialSet, "
            "PurchaseOrderSet, SalesOrderSet, and others. "
            "Supports $filter (e.g. \"CountryCode eq 'DE'\"), $select, $top, $skip."
        ),
        parameters=[
            ToolParameter(
                name="entity_set",
                type="string",
                description="OData entity set name, e.g. 'VendorSet' or 'A_Supplier'",
                required=True,
            ),
            ToolParameter(
                name="filter",
                type="string",
                description="OData $filter expression, e.g. \"CountryCode eq 'UK'\"",
                required=False,
            ),
            ToolParameter(
                name="select",
                type="string",
                description="Comma-separated fields to return via $select",
                required=False,
            ),
            ToolParameter(
                name="top",
                type="integer",
                description=f"Max records to return (default 20, hard cap {_MAX_TOP})",
                required=False,
                default=20,
            ),
            ToolParameter(
                name="skip",
                type="integer",
                description="Records to skip for pagination",
                required=False,
                default=0,
            ),
        ],
        source=ToolSource.NATIVE,
        handler=_query,
        metadata={
            "base_url": _base,
            "allowed_entity_sets": sorted(_allowed) if _allowed else None,
            # auth intentionally omitted — contains credentials
        },
    )
