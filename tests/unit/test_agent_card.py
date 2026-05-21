"""Tests for AgentCard and AgentSkill parsing, serialisation, and edge cases."""

from kazi.agents.agent_card import AgentCard, AgentSkill

FULL_CARD = {
    "name": "summarizer",
    "description": "Summarises documents",
    "version": "2.1",
    "capabilities": ["text-processing", "summarisation"],
    "skills": [
        {
            "name": "summarize",
            "description": "Summarise a document",
            "input_schema": {
                "properties": {"text": {"type": "string", "description": "Document text"}},
                "required": ["text"],
            },
            "output_schema": {
                "properties": {"summary": {"type": "string"}},
            },
        },
        {
            "name": "bullet_points",
            "description": "Extract bullet points",
            "input_schema": {
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    ],
    "authentication": {"type": "bearer", "token": "tok-abc"},
    "metadata": {"owner": "team-nlp"},
}


# ── from_dict parsing ─────────────────────────────────────────────────────────

def test_from_dict_full_card():
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    assert card.name == "summarizer"
    assert card.description == "Summarises documents"
    assert card.version == "2.1"
    assert card.url == "https://agents.example.com"
    assert card.capabilities == ["text-processing", "summarisation"]
    assert len(card.skills) == 2
    assert card.authentication == {"type": "bearer", "token": "tok-abc"}
    assert card.metadata == {"owner": "team-nlp"}


def test_from_dict_skill_fields():
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    summarize = card.skills[0]
    assert summarize.name == "summarize"
    assert summarize.description == "Summarise a document"
    assert "text" in summarize.input_schema.get("properties", {})
    assert "summary" in summarize.output_schema.get("properties", {})


def test_from_dict_minimal_card():
    """Only the required 'name' field present — everything else uses defaults."""
    card = AgentCard.from_dict({"name": "bare-agent"}, url="http://localhost:8000")
    assert card.name == "bare-agent"
    assert card.description == ""
    assert card.version == "1.0"
    assert card.capabilities == []
    assert card.skills == []
    assert card.authentication is None
    assert card.metadata == {}


def test_from_dict_accepts_input_schema_camel_case():
    """Agent cards may use 'inputSchema' (camelCase) instead of 'input_schema'."""
    card_data = {
        "name": "agent",
        "skills": [
            {
                "name": "do_thing",
                "description": "Does a thing",
                "inputSchema": {"properties": {"x": {"type": "string"}}, "required": ["x"]},
            }
        ],
    }
    card = AgentCard.from_dict(card_data, url="http://localhost")
    assert "x" in card.skills[0].input_schema.get("properties", {})


def test_from_dict_prefers_snake_case_over_camel():
    """input_schema wins over inputSchema when both are present."""
    card_data = {
        "name": "agent",
        "skills": [
            {
                "name": "skill",
                "description": "",
                "input_schema": {"properties": {"snake": {}}, "required": []},
                "inputSchema": {"properties": {"camel": {}}, "required": []},
            }
        ],
    }
    card = AgentCard.from_dict(card_data, url="http://localhost")
    props = card.skills[0].input_schema.get("properties", {})
    assert "snake" in props
    assert "camel" not in props


def test_from_dict_no_skills():
    card = AgentCard.from_dict({"name": "no-skills"}, url="http://a.b")
    assert card.skills == []


# ── to_dict serialisation ─────────────────────────────────────────────────────

def test_to_dict_round_trips_name_and_version():
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    d = card.to_dict()
    assert d["name"] == "summarizer"
    assert d["version"] == "2.1"


def test_to_dict_includes_skills():
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    d = card.to_dict()
    assert len(d["skills"]) == 2
    skill_names = [s["name"] for s in d["skills"]]
    assert "summarize" in skill_names
    assert "bullet_points" in skill_names


def test_to_dict_skill_has_expected_keys():
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    skill_dict = card.to_dict()["skills"][0]
    assert "name" in skill_dict
    assert "description" in skill_dict
    assert "input_schema" in skill_dict
    assert "output_schema" in skill_dict


def test_to_dict_does_not_include_authentication():
    """Authentication tokens should not be re-serialised into the card dict."""
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    d = card.to_dict()
    assert "authentication" not in d


def test_to_dict_url_not_included():
    """url is set at discovery time and not re-exported in to_dict."""
    card = AgentCard.from_dict(FULL_CARD, url="https://agents.example.com")
    d = card.to_dict()
    assert "url" not in d


def test_to_dict_includes_capabilities():
    card = AgentCard.from_dict(FULL_CARD, url="https://a.b")
    assert card.to_dict()["capabilities"] == ["text-processing", "summarisation"]


# ── AgentSkill ────────────────────────────────────────────────────────────────

def test_agent_skill_defaults():
    skill = AgentSkill(name="do_it", description="Does it")
    assert skill.input_schema == {}
    assert skill.output_schema == {}


def test_agent_skill_with_schemas():
    skill = AgentSkill(
        name="query",
        description="Query data",
        input_schema={"properties": {"q": {"type": "string"}}},
        output_schema={"properties": {"result": {"type": "string"}}},
    )
    assert "q" in skill.input_schema["properties"]
    assert "result" in skill.output_schema["properties"]
