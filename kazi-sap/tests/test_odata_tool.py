"""
Unit tests for sap_odata_tool.

Uses respx to mock HTTP so no real SAP endpoint is required.
"""
from __future__ import annotations

import httpx
import pytest
import respx
from kazi_sap.auth import APIKeyAuth
from kazi_sap.tools import sap_odata_tool

BASE = "https://sandbox.api.sap.com/test/odata/v2/API_BUSINESS_PARTNER"


@pytest.fixture
def auth():
    return APIKeyAuth(api_key="test-key")


@pytest.fixture
def tool(auth):
    return sap_odata_tool(
        BASE,
        auth,
        allowed_entity_sets=["VendorSet", "CustomerSet"],
    )


@pytest.fixture
def vendor_page():
    return {
        "d": {
            "results": [
                {"VendorId": "V001", "Name": "Acme", "CountryCode": "GB"},
                {"VendorId": "V002", "Name": "Global Parts", "CountryCode": "DE"},
            ]
        }
    }


class TestAllowList:
    @pytest.mark.asyncio
    async def test_blocked_entity_set(self, tool):
        result = await tool.handler(entity_set="InternalSalaries")
        assert "not in the allowed list" in result
        assert "VendorSet" in result

    @pytest.mark.asyncio
    async def test_allowed_entity_set_passes(self, tool, vendor_page):
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(200, json=vendor_page)
            )
            result = await tool.handler(entity_set="VendorSet")
        assert "VendorSet" in result
        assert "V001" in result


class TestTopCap:
    @pytest.mark.asyncio
    async def test_top_capped_at_500(self, tool, vendor_page):
        captured_params: list[dict] = []

        def _inspect(request: httpx.Request) -> httpx.Response:
            captured_params.append(dict(
                kv.split("=", 1) for kv in request.url.query.decode().split("&") if "=" in kv
            ))
            return httpx.Response(200, json=vendor_page)

        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(side_effect=_inspect)
            await tool.handler(entity_set="VendorSet", top=9999)

        top_val = int(captured_params[0].get("%24top", captured_params[0].get("$top", 0)))
        assert top_val <= 500


class TestResponseFormats:
    @pytest.mark.asyncio
    async def test_odata_v2_envelope(self, tool):
        payload = {"d": {"results": [{"VendorId": "V001", "Name": "Acme"}]}}
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = await tool.handler(entity_set="VendorSet")
        assert "V001" in result

    @pytest.mark.asyncio
    async def test_odata_v4_envelope(self, tool):
        payload = {"value": [{"VendorId": "V001", "Name": "Acme"}]}
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = await tool.handler(entity_set="VendorSet")
        assert "V001" in result

    @pytest.mark.asyncio
    async def test_empty_result(self, tool):
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(200, json={"d": {"results": []}})
            )
            result = await tool.handler(entity_set="VendorSet")
        assert "No records" in result

    @pytest.mark.asyncio
    async def test_http_error_returned_as_string(self, tool):
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(401, text="Unauthorized")
            )
            result = await tool.handler(entity_set="VendorSet")
        assert "401" in result

    @pytest.mark.asyncio
    async def test_timeout_returned_as_string(self, auth):
        slow_tool = sap_odata_tool(BASE, auth, timeout=1)
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                side_effect=httpx.TimeoutException("timed out")
            )
            result = await slow_tool.handler(entity_set="VendorSet")
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_metadata_fields_stripped(self, tool):
        payload = {
            "d": {
                "results": [
                    {
                        "VendorId": "V001",
                        "__metadata": {"type": "API_BUSINESS_PARTNER.A_SupplierType"},
                        "__deferred": {"uri": "VendorSet('V001')/ToPartnerFunctions"},
                        "Name": "Acme",
                    }
                ]
            }
        }
        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = await tool.handler(entity_set="VendorSet")
        assert "__metadata" not in result
        assert "__deferred" not in result
        assert "Acme" in result


class TestAuth:
    @pytest.mark.asyncio
    async def test_api_key_header_sent(self, tool, vendor_page):
        seen_headers: list[dict] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            seen_headers.append(dict(request.headers))
            return httpx.Response(200, json=vendor_page)

        with respx.mock:
            respx.get(url__regex=r".*/VendorSet.*").mock(side_effect=_capture)
            await tool.handler(entity_set="VendorSet")

        assert seen_headers[0].get("apikey") == "test-key"
