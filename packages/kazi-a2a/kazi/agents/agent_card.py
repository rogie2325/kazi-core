from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentSkill:
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)


@dataclass
class AgentCard:
    """
    A2A Agent Card — the public capability manifest of a remote agent.
    Fetched from <agent_url>/.well-known/agent.json
    """

    name: str
    description: str
    url: str
    version: str = "1.0"
    capabilities: list[str] = field(default_factory=list)
    skills: list[AgentSkill] = field(default_factory=list)
    authentication: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, url: str) -> AgentCard:
        skills = [
            AgentSkill(
                name=s["name"],
                description=s.get("description", ""),
                input_schema=s.get("input_schema", s.get("inputSchema", {})),
                output_schema=s.get("output_schema", s.get("outputSchema", {})),
            )
            for s in data.get("skills", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            url=url,
            version=data.get("version", "1.0"),
            capabilities=data.get("capabilities", []),
            skills=skills,
            authentication=data.get("authentication"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema,
                    "output_schema": s.output_schema,
                }
                for s in self.skills
            ],
            "metadata": self.metadata,
        }
