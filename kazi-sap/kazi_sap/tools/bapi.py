"""
SAP BAPI / RFC tool for kazi.

Wraps pyrfc to call any SAP function module (BAPI or RFC) from an LLM
tool call.  Designed with two layers of safety:

1. allowed_bapis whitelist    — the LLM can only call explicitly permitted
                                function modules; any other name is rejected
                                before a connection is opened.
2. testrun_by_default=True    — injects TESTRUN='X' into every call so SAP
                                validates the request and returns the RETURN
                                table without committing.  The caller must
                                explicitly pass testrun=False to commit, which
                                should only happen after human approval via
                                kazi.run_with_approval().

Requires::

    pip install kazi-sap[bapi]   # pulls pyrfc
    # Also requires SAP NW RFC SDK (nwrfcsdk) installed on the host —
    # see https://github.com/SAP/PyRFC for setup instructions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

logger = logging.getLogger(__name__)

# SAP function module names: uppercase letters, digits, underscores, 1–61 chars.
_VALID_FM_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,60}$")


def sap_bapi_tool(
    conn_params: dict,
    *,
    allowed_bapis: list[str] | None = None,
    testrun_by_default: bool = True,
) -> ToolDefinition:
    """
    Return a ToolDefinition for calling SAP BAPIs and RFC function modules.

    conn_params
        pyrfc connection parameters dict::

            {
                "ashost": "10.0.0.1",
                "sysnr":  "00",
                "client": "100",
                "user":   "RFC_USER",
                "passwd": "...",    # use SecretRef in production
                "lang":   "EN",
            }

        conn_params is captured in the closure but intentionally omitted from
        tool metadata so credentials are never serialised to logs or responses.

    allowed_bapis
        Allowlist of BAPI/RFC names the LLM may call.
        Strongly recommended — without it the LLM can call any RFC on the system,
        including dangerous ones like SUSR_USER_CHANGE_PASSWORD_RFC.
        Example: ["BAPI_VENDOR_CHANGE", "BAPI_CUSTOMER_CHANGE", "BAPI_TRANSACTION_COMMIT"]

    testrun_by_default
        When True (default), TESTRUN='X' is automatically added to every call.
        SAP validates the full transaction and returns the RETURN table, but
        nothing is committed to the database.
        Set to False only if every call to this tool should commit immediately
        (rare — usually you want the approval gate instead).
    """
    _allowed = set(allowed_bapis) if allowed_bapis else None

    async def _call(
        bapi_name: str,
        params: str = "{}",
        testrun: bool = testrun_by_default,
    ) -> str:
        # 1. Name format validation
        if not _VALID_FM_RE.match(bapi_name):
            return (
                f"Invalid function module name: '{bapi_name}'. "
                "SAP names must match [A-Z][A-Z0-9_]{0,60}."
            )

        # 2. Allowlist check
        if _allowed is not None and bapi_name not in _allowed:
            return (
                f"BAPI '{bapi_name}' is not in the allowed list. "
                f"Permitted: {sorted(_allowed)}"
            )

        # 3. Parse import parameters
        try:
            args: dict = json.loads(params) if params.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Invalid JSON in params: {exc}"

        # 4. Inject TESTRUN flag
        if testrun:
            args["TESTRUN"] = "X"
            logger.info("BAPI %s called in TESTRUN mode (no commit)", bapi_name)
        else:
            logger.warning("BAPI %s called with testrun=False — changes WILL be committed", bapi_name)

        # 5. Execute via pyrfc in a thread (pyrfc is synchronous)
        def _sync_call() -> dict:
            try:
                import pyrfc  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "pyrfc is required for BAPI calls. "
                    "Install: pip install kazi-sap[bapi]  "
                    "(also requires SAP NW RFC SDK — see https://github.com/SAP/PyRFC)"
                )
            with pyrfc.Connection(**conn_params) as conn:
                return conn.call(bapi_name, **args)

        try:
            result = await asyncio.to_thread(_sync_call)
        except Exception as exc:
            return f"BAPI error calling {bapi_name}: {exc}"

        # 6. Format output — lead with the RETURN table for clear status
        return_msgs = result.get("RETURN", [])
        lines: list[str] = []
        prefix = "[TESTRUN — not committed]" if testrun else "[COMMITTED]"
        lines.append(prefix)

        if return_msgs:
            for msg in return_msgs:
                t = msg.get("TYPE", "?")
                text = msg.get("MESSAGE", "").strip()
                lines.append(f"  [{t}] {text}")
        else:
            lines.append("  (no RETURN messages)")

        # Include other scalar output fields for context
        extra = {
            k: v for k, v in result.items()
            if k != "RETURN" and not isinstance(v, (list, dict))
        }
        if extra:
            lines.append(f"  Output fields: {extra}")

        return "\n".join(lines)

    return ToolDefinition(
        name="sap_bapi",
        description=(
            "Call a SAP BAPI or RFC function module to read or write SAP data. "
            "Pass import parameters as a JSON string. "
            "By default runs in TESTRUN mode — SAP validates but does not commit. "
            "Only set testrun=false after receiving explicit human approval."
        ),
        parameters=[
            ToolParameter(
                name="bapi_name",
                type="string",
                description="BAPI or RFC name in uppercase, e.g. 'BAPI_VENDOR_CHANGE'",
                required=True,
            ),
            ToolParameter(
                name="params",
                type="string",
                description="JSON-encoded import parameters for the function module",
                required=False,
                default="{}",
            ),
            ToolParameter(
                name="testrun",
                type="boolean",
                description=(
                    "Dry-run mode: SAP validates the call but does not commit (default true). "
                    "Set false only after human approval."
                ),
                required=False,
                default=testrun_by_default,
            ),
        ],
        source=ToolSource.NATIVE,
        handler=_call,
        metadata={
            "testrun_by_default": testrun_by_default,
            "allowed_bapis": sorted(_allowed) if _allowed else None,
            # conn_params intentionally omitted — contains credentials
        },
    )
