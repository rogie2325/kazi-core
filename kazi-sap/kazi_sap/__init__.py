"""
kazi-sap — SAP connector and data scrubber for kazi.

Adds three capabilities on top of kazi:

  1. OData query tool    — read any SAP entity set via REST
  2. BAPI / RFC tool     — call SAP function modules (with testrun safety)
  3. Data scrub pipeline — LLM-driven master data quality analysis
                           and approval-gated remediation

Quick start::

    from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider
    from kazi_sap import SAPScrubPipeline, ScrubConfig
    from kazi_sap.auth import APIKeyAuth
    from kazi_sap.tools import sap_odata_tool, sap_bapi_tool

    auth = APIKeyAuth(api_key="your-sap-api-key")
    base_url = "https://sandbox.api.sap.com/.../API_BUSINESS_PARTNER"

    config = KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"))

    async with await Kazi.create(config) as kazi:
        kazi.add_tool(sap_odata_tool(base_url, auth, allowed_entity_sets=["VendorSet"]))
        kazi.add_tool(sap_bapi_tool(conn_params, allowed_bapis=["BAPI_VENDOR_CHANGE"]))

        pipeline = SAPScrubPipeline(kazi=kazi, auth=auth, base_url=base_url)

        results = await pipeline.analyze("VendorSet", filter="CountryCode eq 'UK'")
        print(pipeline.report_text(results))
"""

from kazi_sap.auth import APIKeyAuth, BasicAuth, SAPAuth, XSUAAAuth
from kazi_sap.models import RemediationReport, ScrubIssue, ScrubResult
from kazi_sap.scrubber import SAPScrubPipeline, ScrubConfig
from kazi_sap.tools import sap_bapi_tool, sap_odata_tool

__version__ = "0.1.0"

__all__ = [
    # Auth
    "SAPAuth",
    "APIKeyAuth",
    "BasicAuth",
    "XSUAAAuth",
    # Models
    "ScrubIssue",
    "ScrubResult",
    "RemediationReport",
    # Pipeline
    "SAPScrubPipeline",
    "ScrubConfig",
    # Tools
    "sap_odata_tool",
    "sap_bapi_tool",
]
