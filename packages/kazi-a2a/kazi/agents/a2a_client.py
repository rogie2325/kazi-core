from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Optional

import httpx

from kazi.agents.agent_card import AgentCard, AgentSkill
from kazi.core.config import A2AConfig
from kazi.core.exceptions import A2AConnectionError, A2ATimeoutError, AgentNotFoundError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource, ToolRegistry
from kazi.core.security import SecurityConfig

logger = logging.getLogger(__name__)


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
        security: Optional[SecurityConfig] = None,
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
        for skill in card.skills:
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

        try:
            resp = await self._http.post(
                card.url.rstrip("/") + "/a2a",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        except asyncio.TimeoutError:
            raise A2ATimeoutError(f"Task delegation to '{agent_name}' timed out")
        except httpx.RequestError as exc:
            raise A2AConnectionError(f"Cannot reach agent '{agent_name}': {exc}") from exc

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
    ) -> str:
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
            except Exception:
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
