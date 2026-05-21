"""
Pydantic models for SAP data scrubbing results.

These are used as response_schema= in kazi.run() calls so the LLM
returns structured, validated JSON rather than free text.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScrubIssue(BaseModel):
    """A single data quality problem found in one SAP field."""

    field: str = Field(description="SAP field name, e.g. 'CountryCode' or 'Street'")
    current: str = Field(description="The current value stored in SAP")
    issue: str = Field(description="Plain-English description of the data quality problem")
    suggested: str | None = Field(
        None,
        description="Corrected value to apply, or null when the fix is ambiguous",
    )


class ScrubResult(BaseModel):
    """
    LLM analysis result for a single SAP record.

    Returned as structured output from the scrubber's analysis step.
    The LLM populates this via response_schema= so it is always valid JSON
    with typed fields — no free-text parsing required.
    """

    record_id: str = Field(description="SAP record primary key, e.g. vendor ID or material number")
    entity_type: str = Field(description="SAP entity type, e.g. 'Vendor', 'Customer', 'Material'")
    issues: list[ScrubIssue] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = Field(
        description=(
            "low = cosmetic only (whitespace, casing); "
            "medium = functional impact (wrong code, bad format); "
            "high = financial or compliance risk (duplicate, invalid bank details)"
        )
    )
    auto_fixable: bool = Field(
        description=(
            "True only when every issue has an unambiguous suggested value "
            "AND severity is not 'high'"
        )
    )
    notes: str | None = Field(None, description="Extra context or caveats, or null")


class RemediationReport(BaseModel):
    """Summary produced after a remediate() run."""

    entity_set: str
    total_analyzed: int
    clean: int
    with_issues: int
    auto_fixed: int
    pending_approval: int
    skipped_high_severity: int
    errors: int
    results: list[ScrubResult]
