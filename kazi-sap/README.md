# kazi-sap

SAP connector and LLM-driven data scrubber for [kazi](../README.md).

Adds three capabilities on top of the kazi orchestrator:

| Capability | What it does |
|---|---|
| **OData tool** | Read any SAP entity set via REST — the LLM can filter, paginate, and select fields |
| **BAPI / RFC tool** | Call SAP function modules; testrun-safe by default, writes only with human approval |
| **Scrub pipeline** | Fetch → LLM quality analysis → structured report → approval-gated write-back |

---

## Install

```bash
pip install kazi-sap                  # OData tool + scrub pipeline
pip install "kazi-sap[bapi]"          # + BAPI/RFC support (requires SAP NW RFC SDK)
pip install "kazi-sap[bapi,dev]"      # + test dependencies
```

`kazi-sap[bapi]` pulls `pyrfc`, which requires the [SAP NW RFC SDK](https://github.com/SAP/PyRFC) installed on the host separately.

---

## Authentication

Three strategies cover the full range of SAP deployment types:

```python
from kazi_sap.auth import APIKeyAuth, BasicAuth, XSUAAAuth

# SAP API Business Hub / BTP managed APIs
auth = APIKeyAuth(api_key=os.environ["SAP_API_KEY"])

# On-premise SAP via HTTP Basic
auth = BasicAuth(username="RFC_USER", password=os.environ["SAP_PASS"])

# BTP OAuth2 via XSUAA — token is fetched and cached automatically,
# refreshed 30s before expiry
auth = XSUAAAuth(
    client_id=os.environ["XSUAA_CLIENT_ID"],
    client_secret=os.environ["XSUAA_CLIENT_SECRET"],
    token_url="https://<subdomain>.authentication.eu10.hana.ondemand.com/oauth/token",
)
```

All strategies wrap credentials in `SecretRef` so they never appear in `repr()` or logs.

---

## Tools

### OData query tool

```python
from kazi import Kazi, KaziConfig
from kazi_sap.tools import sap_odata_tool
from kazi_sap.auth import APIKeyAuth

auth = APIKeyAuth(api_key=os.environ["SAP_API_KEY"])
base_url = "https://sandbox.api.sap.com/.../API_BUSINESS_PARTNER"

async with await Kazi.create(KaziConfig(...)) as kazi:
    kazi.add_tool(sap_odata_tool(
        base_url,
        auth,
        allowed_entity_sets=["VendorSet", "CustomerSet"],  # whitelist
    ))

    result = await kazi.run("List all UK vendors with missing postal codes")
```

The tool exposes `$filter`, `$select`, `$top`, and `$skip` to the LLM. `$top` is hard-capped at 500 regardless of what the LLM requests. `allowed_entity_sets=None` allows all entity sets — not recommended in production.

Supports both OData v2 (`d.results`) and v4 (`value`) response envelopes.

### BAPI / RFC tool

```python
from kazi_sap.tools import sap_bapi_tool

conn_params = {
    "ashost": "10.0.0.1", "sysnr": "00", "client": "100",
    "user": "RFC_USER", "passwd": os.environ["SAP_PASS"], "lang": "EN",
}

kazi.add_tool(sap_bapi_tool(
    conn_params,
    allowed_bapis=["BAPI_VENDOR_CHANGE", "BAPI_TRANSACTION_COMMIT"],
))
```

By default every call runs with `TESTRUN='X'` — SAP validates the full transaction and returns the `RETURN` table without committing. The LLM must explicitly pass `testrun=false`, which should only happen inside a `kazi.run_with_approval()` call after a human approval gate.

`allowed_bapis` is strongly recommended. Without it the LLM can call any RFC on the system, including dangerous ones like `SUSR_USER_CHANGE_PASSWORD_RFC`.

---

## Data scrub pipeline

`SAPScrubPipeline` implements a three-phase read → analyze → write-back workflow.

```python
from kazi_sap import SAPScrubPipeline, ScrubConfig

pipeline = SAPScrubPipeline(
    kazi=kazi,
    auth=auth,
    base_url=base_url,
    config=ScrubConfig(
        concurrency=5,              # parallel LLM calls during analysis
        auto_fix_severity="low",    # only auto-apply low-severity fixes
        page_size=50,
        max_records=500,
    ),
)
```

### Phase 1 — Analyze (read-only, safe against production)

```python
results: list[ScrubResult] = await pipeline.analyze(
    "VendorSet",
    filter="CountryCode eq 'UK'",
)
```

Fetches records via OData, sends each to the LLM with an entity-specific prompt, and returns a structured `ScrubResult` per record. No SAP writes occur.

### Phase 2 — Report

```python
print(pipeline.report_text(results))
```

```
SAP Data Quality Report — 42 record(s) analysed
============================================================

[HIGH] — 2 record(s)
  V-00123 (Vendor) [manual review]
    • BankAccount: invalid IBAN format → (no suggestion)

[MEDIUM] — 7 record(s)
  V-00456 (Vendor) [auto-fixable]
    • CountryCode: 'UK' → 'GB'

[LOW] — 12 record(s)
  ...

✓ Clean records: 21
  Total issues: 31  |  High: 2  Medium: 7  Low: 12
```

### Phase 3 — Remediate (approval-gated)

```python
async def approve(tool_calls):
    print(tool_calls)
    return tool_calls if input("Approve? [y/n]: ") == "y" else None

report: RemediationReport = await pipeline.remediate(
    results,
    approval_callback=approve,
    bapi_name="BAPI_VENDOR_CHANGE",
)
print(f"Fixed: {report.auto_fixed}  Pending: {report.pending_approval}  "
      f"Skipped (high): {report.skipped_high_severity}  Errors: {report.errors}")
```

Remediation rules:
- **Clean records** — skipped
- **High severity** — always skipped; logged for manual review (never auto-applied)
- **`auto_fixable=False`** — left as pending
- **Severity above `auto_fix_severity` threshold** — left as pending
- **Everything else** — proposed to `approval_callback`; committed only if callback returns non-None

### Structured output models

```python
from kazi_sap.models import ScrubIssue, ScrubResult, RemediationReport

# ScrubIssue: field, current, issue, suggested
# ScrubResult: record_id, entity_type, issues, severity, auto_fixable, notes
# RemediationReport: total_analyzed, clean, with_issues, auto_fixed,
#                    pending_approval, skipped_high_severity, errors, results
```

The LLM populates `ScrubResult` via `response_schema=` — structured JSON with typed fields, no free-text parsing.

---

## End-to-end example

```bash
export SAP_BASE_URL="https://sandbox.api.sap.com/.../API_BUSINESS_PARTNER"
export SAP_API_KEY="your-sap-key"
export ANTHROPIC_API_KEY="sk-ant-..."

python examples/vendor_cleanup.py
python examples/vendor_cleanup.py --filter "CountryCode eq 'DE'" --max-records 100
python examples/vendor_cleanup.py --dry-run   # analyze + report only, no writes
```

---

## Security model

| Concern | Mitigation |
|---|---|
| LLM reading unauthorized entity sets | `allowed_entity_sets` whitelist on `sap_odata_tool` |
| LLM calling dangerous BAPIs | `allowed_bapis` whitelist on `sap_bapi_tool` |
| Accidental SAP writes | `testrun_by_default=True` on every BAPI call |
| Writes without human review | All `remediate()` writes go through `approval_callback` |
| High-severity auto-remediation | Hard-coded skip — high severity always requires manual review |
| Credentials in logs | All secrets wrapped in `SecretRef`; `conn_params` omitted from tool metadata |
| Full-table OData scans | `$top` hard-capped at 500; `max_records` cap in pipeline fetch |

---

## Development

```bash
cd kazi-sap
pip install -e ".[bapi,dev]"

pytest                          # unit + mock tests
pytest -m "not llm"             # skip live LLM tests (default)
pytest -m "not integration"     # skip SAP sandbox tests
pytest tests/test_scrubber.py   # single file
```

Test markers defined in `pyproject.toml`:
- `llm` — requires a live LLM (slow, costs money)
- `integration` — requires a SAP sandbox connection
