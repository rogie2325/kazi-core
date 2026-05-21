"""
SAP authentication strategies.

Three strategies cover the full range of SAP deployment types:

  APIKeyAuth   — SAP API Business Hub / BTP managed APIs (header-based)
  BasicAuth    — On-premise SAP systems via HTTP Basic (legacy, still common)
  XSUAAAuth    — OAuth2 client credentials via SAP BTP XSUAA service

All strategies use SecretRef internally so credentials never appear in
repr() output or log statements.

Usage::

    from kazi_sap.auth import APIKeyAuth, XSUAAAuth, BasicAuth

    # BTP / API Hub
    auth = APIKeyAuth(api_key=os.environ["SAP_API_KEY"])

    # On-premise
    auth = BasicAuth(username="RFC_USER", password=os.environ["SAP_PASS"])

    # BTP OAuth2
    auth = XSUAAAuth(
        client_id=os.environ["XSUAA_CLIENT_ID"],
        client_secret=os.environ["XSUAA_CLIENT_SECRET"],
        token_url="https://<subdomain>.authentication.eu10.hana.ondemand.com/oauth/token",
    )
"""
from __future__ import annotations

import base64
import logging
import time
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class SAPAuth(ABC):
    """Base class for SAP authentication strategies."""

    @abstractmethod
    def sync_headers(self) -> dict[str, str]:
        """Return auth headers synchronously (used when no async context is available)."""
        ...

    async def headers(self) -> dict[str, str]:
        """Return auth headers. Default delegates to sync_headers."""
        return self.sync_headers()


class APIKeyAuth(SAPAuth):
    """
    SAP API Business Hub / BTP managed API key authentication.

    Passes the key as the ``APIKey`` header, which is the standard for
    SAP sandbox and production APIs on api.sap.com.
    """

    def __init__(self, api_key: str) -> None:
        from kazi.core.secrets import SecretRef
        self._key = SecretRef.coerce(api_key)

    def sync_headers(self) -> dict[str, str]:
        return {"APIKey": self._key.resolve() or ""}


class BasicAuth(SAPAuth):
    """
    HTTP Basic authentication for on-premise SAP systems.

    Credentials are Base64-encoded at header-construction time, not at
    construction time, so the password stays in SecretRef until needed.
    """

    def __init__(self, username: str, password: str) -> None:
        from kazi.core.secrets import SecretRef
        self._username = username
        self._password = SecretRef.coerce(password)

    def sync_headers(self) -> dict[str, str]:
        creds = f"{self._username}:{self._password.resolve() or ''}"
        encoded = base64.b64encode(creds.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}


class XSUAAAuth(SAPAuth):
    """
    OAuth2 client credentials flow via SAP BTP XSUAA.

    Fetches and caches a bearer token; refreshes automatically 30 s before
    expiry so in-flight requests never hit an expired token mid-stream.

    token_url example:
        https://<subdomain>.authentication.eu10.hana.ondemand.com/oauth/token
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
    ) -> None:
        from kazi.core.secrets import SecretRef
        self._client_id = client_id
        self._client_secret = SecretRef.coerce(client_secret)
        self._token_url = token_url
        self._token: str | None = None
        self._expires_at: float = 0.0

    def sync_headers(self) -> dict[str, str]:
        # Sync path can only use a previously cached token.
        if self._token and time.monotonic() < self._expires_at:
            return {"Authorization": f"Bearer {self._token}"}
        raise RuntimeError(
            "XSUAAAuth: no cached token available — call `await auth.headers()` "
            "at least once before using sync_headers()."
        )

    async def headers(self) -> dict[str, str]:
        # Refresh 30 s before expiry so we never hand a near-expired token to httpx.
        if not self._token or time.monotonic() >= self._expires_at - 30:
            await self._refresh()
        return {"Authorization": f"Bearer {self._token}"}

    async def _refresh(self) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret.resolve() or "",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self._expires_at = time.monotonic() + expires_in
        logger.debug("XSUAA token refreshed — expires in %ds", expires_in)
