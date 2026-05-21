from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import random
import uuid
from urllib.parse import urlparse

import httpx

from kazi.agents.agent_card import AgentCard, AgentSkill
from kazi.core.config import A2AConfig
from kazi.core.exceptions import A2AConnectionError, A2ATimeoutError, AgentNotFoundError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolRegistry, ToolSource
from kazi.core.security import SecurityConfig

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_SKILLS_PER_AGENT = 50


def _validate_agent_url(url: str, allowed_hosts: list[str]) -> None:
    """
    Basic SSRF guard applied before every agent card fetch.

    Blocks:
      - Non-HTTP(S) schemes (file://, ftp://, etc.)
      - URLs with no hostname
      - IP-literal hostnames in private / loopback / reserved ranges
      - Hostnames not in allowed_hosts (when the list is non-empty)

    Limitation: DNS-rebinding attacks (hostname resolves to a public IP at
    check time, then switches to a private IP at connect time) are NOT caught
    here. For multi-tenant production, add a network-level egress firewall
    or use an HTTP proxy with IP-based blocking.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise A2AConnectionError(f"Malformed agent URL {url!r}: {exc}") from exc

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise A2AConnectionError(
            f"Agent URL scheme {parsed.scheme!r} not allowed — must be http or https"
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise A2AConnectionError(f"Agent URL has no hostname: {url!r}")

    # Block IP-literal private / loopback / reserved addresses
    try:
        ip = ipaddress.ip_address(hostname)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise A2AConnectionError(
                f"Agent URL {url!r} resolves to an internal address — SSRF blocked"
            )
    except ValueError:
        pass  # not an IP literal; domain name — fine

    # Allowlist check (empty list = allow all non-private hosts)
    if allowed_hosts:
        match = any(
            hostname == h or hostname.endswith("." + h)
            for h in allowed_hosts
        )
        if not match:
            raise A2AConnectionError(
                f"Agent host {hostname!r} is not in the allowed_hosts allowlist"
            )


class A2ABridge:
    """
    Discovers remote A2A agents via Agent Cards and registers their
    skills as ToolDefinitions in the shared registry.

    Security hardening
    ──────────────────
    - TLS verification is ON by default (security.verify_tls).
      Set verify_tls=False only for internal dev environments with self-signed certs.
    - All delegation results are passed through ContentPolicy.wrap() so they
      are tagged as external content before entering the LLM context.
    - Agent Card fetches time out at 10 s; task delegation respects
      A2AConfig.delegation_timeout.
    """

    def __init__(
        self,
        config: A2AConfig,
        registry: ToolRegistry,
        security: SecurityConfig | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.security = security or SecurityConfig()
        self._agents: dict[str, AgentCard] = {}
        self._http = httpx.AsyncClient(
            timeout=config.delegation_timeout,
            verify=self.security.verify_tls,
        )

    # ── discovery ─────────────────────────────────────────────────────────

    async def discover_agents(self) -> list[AgentCard]:
        cards: list[AgentCard] = []
        for endpoint in self.config.discovery_endpoints:
            try:
                card = await self._fetch_card(endpoint)
                self._agents[card.name] = card
                self._register_skills(card)
                cards.append(card)
                logger.info(
                    "Discovered A2A agent '%s' with %d skills at %s",
                    card.name, len(card.skills), endpoint,
                )
            except Exception as exc:
                logger.error("Failed to discover agent at %s: %s", endpoint, exc)
        return cards

    async def register_agent(self, url: str) -> AgentCard:
        card = await self._fetch_card(url)
        self._agents[card.name] = card
        self._register_skills(card)
        return card

    async def _fetch_card(self, url: str) -> AgentCard:
        _validate_agent_url(url, self.config.allowed_hosts)
        card_url = url.rstrip("/") + "/.well-known/agent.json"
        try:
            # Short timeout for card discovery — we don't want slow agents to
            # block startup
            resp = await asyncio.wait_for(
                self._http.get(card_url),
                timeout=10.0,
            )
            resp.raise_for_status()
        except asyncio.TimeoutError:
            raise A2AConnectionError(f"Agent card fetch timed out: {card_url}")
        except httpx.RequestError as exc:
            raise A2AConnectionError(f"Cannot reach agent at {card_url}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise A2AConnectionError(
                f"Agent card at {card_url} returned HTTP {exc.response.status_code}"
            ) from exc
        return AgentCard.from_dict(resp.json(), url)

    def _register_skills(self, card: AgentCard) -> None:
        skills = card.skills
        if len(skills) > _MAX_SKILLS_PER_AGENT:
            logger.warning(
                "Agent '%s' advertises %d skills — capping at %d to prevent registry bloat",
                card.name, len(skills), _MAX_SKILLS_PER_AGENT,
            )
            skills = skills[:_MAX_SKILLS_PER_AGENT]
        for skill in skills:
            params = self._skill_params(skill)
            agent_name = card.name
            skill_name = skill.name

            async def handler(agent=agent_name, sk=skill_name, **kwargs):
                return await self.delegate(agent, sk, kwargs)

            tool = ToolDefinition(
                name=f"a2a__{card.name}__{skill.name}",
                description=f"[Agent: {card.name}] {skill.description}",
                parameters=params,
                source=ToolSource.A2A,
                handler=handler,
                a2a_agent_url=card.url,
                metadata={"agent_name": card.name, "skill_name": skill.name},
            )
            self.registry.register(tool, category=f"a2a_{card.name}")

    @staticmethod
    def _skill_params(skill: AgentSkill) -> list[ToolParameter]:
        props = skill.input_schema.get("properties", {})
        required = set(skill.input_schema.get("required", []))
        if not props:
            return [ToolParameter(
                name="task_description",
                type="string",
                description="Description of the task to delegate",
                required=True,
            )]
        return [
            ToolParameter(
                name=k,
                type=v.get("type", "string"),
                description=v.get("description", ""),
                required=k in required,
                default=v.get("default"),
            )
            for k, v in props.items()
        ]

    # ── task delegation ───────────────────────────────────────────────────

    async def delegate(self, agent_name: str, skill_name: str, params: dict) -> str:
        from kazi.utils.telemetry import span

        if agent_name not in self._agents:
            raise AgentNotFoundError(f"Unknown agent: {agent_name}")

        card = self._agents[agent_name]
        task_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": task_id,
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{
                        "type": "text",
                        "text": json.dumps({"skill": skill_name, "parameters": params}),
                    }],
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if card.authentication and card.authentication.get("type") == "bearer":
            token = card.authentication.get("token", "")
            headers["Authorization"] = f"Bearer {token}"

        max_attempts = self.config.max_retries + 1
        last_exc: Exception = RuntimeError("unreachable")

        with span("kazi.a2a.delegate", {"agent": agent_name, "skill": skill_name}):
            for attempt in range(max_attempts):
                try:
                    resp = await self._http.post(
                        card.url.rstrip("/") + "/a2a",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    break  # success — exit retry loop

                except asyncio.TimeoutError:
                    raise A2ATimeoutError(f"Task delegation to '{agent_name}' timed out")

                except httpx.HTTPStatusError as exc:
                    # 4xx = client error, not retryable
                    if exc.response.status_code < 500:
                        raise A2AConnectionError(
                            f"Agent '{agent_name}' returned HTTP {exc.response.status_code}"
                        ) from exc
                    last_exc = exc

                except httpx.RequestError as exc:
                    last_exc = exc

                if attempt < max_attempts - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "A2A delegation to '%s' failed (attempt %d/%d): %s — retrying in %.1fs",
                        agent_name, attempt + 1, max_attempts, last_exc, delay,
                    )
                    await asyncio.sleep(delay)
            else:
                raise A2AConnectionError(
                    f"A2A delegation to '{agent_name}' failed after {max_attempts} attempt(s)"
                ) from last_exc

        data = resp.json()
        if "error" in data:
            raw = f"A2A Error from '{agent_name}': {data['error'].get('message', 'unknown')}"
            return self.security.content.wrap(f"a2a__{agent_name}__{skill_name}", raw)

        result = data.get("result", {})
        state = result.get("status", {}).get("state", "unknown")

        if state == "completed":
            raw = self._extract_text(result)
        elif state in ("working", "submitted"):
            raw = await self._poll(card, task_id, headers)
        else:
            raw = f"Task status: {state}"

        # Tag all A2A results as external content
        return self.security.content.wrap(f"a2a__{agent_name}__{skill_name}", raw)

    async def _poll(
        self,
        card: AgentCard,
        task_id: str,
        headers: dict,
        max_polls: int = 30,
        interval: float = 2.0,
        max_consecutive_errors: int = 3,
    ) -> str:
        consecutive_errors = 0
        for _ in range(max_polls):
            await asyncio.sleep(interval)
            payload = {
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "id": f"p-{task_id}",
                "params": {"id": task_id},
            }
            try:
                resp = await self._http.post(
                    card.url.rstrip("/") + "/a2a",
                    json=payload,
                    headers=headers,
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("Poll error for task %s (%d/%d): %s", task_id, consecutive_errors, max_consecutive_errors, exc)
                if consecutive_errors >= max_consecutive_errors:
                    raise A2AConnectionError(
                        f"Agent '{card.name}' unreachable after {consecutive_errors} consecutive poll errors"
                    ) from exc
                continue
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                state = result.get("status", {}).get("state", "unknown")
                if state == "completed":
                    return self._extract_text(result)
                if state == "failed":
                    return f"Remote task failed: {result.get('status', {}).get('message', '')}"
        raise A2ATimeoutError(f"Task polling timed out after {max_polls * interval:.0f}s")

    @staticmethod
    def _extract_text(result: dict) -> str:
        texts: list[str] = []
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part["text"])
        return "\n".join(texts) if texts else "Task completed (no text output)"

    # ── lifecycle ─────────────────────────────────────────────────────────

    def list_agents(self) -> list[AgentCard]:
        return list(self._agents.values())

    async def close(self) -> None:
        await self._http.aclose()
