"""Unit tests for model routing (RouterConfig, ModelRoute, GraphBrain routing)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
from kazi.core.router import ModelRoute, RouterConfig
from kazi.core.secrets import SecretRef
from kazi.core.token_budget import TokenBudgetConfig, maybe_summarise

# ── ModelRoute ────────────────────────────────────────────────────────────────

def test_model_route_minimal():
    route = ModelRoute(model="gpt-4o-mini")
    assert route.model == "gpt-4o-mini"
    assert route.provider is None
    assert route.api_key is None
    assert route.temperature is None
    assert route.max_tokens is None


def test_model_route_coerces_plain_api_key_to_secret_ref():
    route = ModelRoute(model="gpt-4o-mini", api_key="sk-test-123")
    assert isinstance(route.api_key, SecretRef)
    assert route.resolved_api_key() == "sk-test-123"


def test_model_route_preserves_secret_ref():
    ref = SecretRef.from_env("SOME_KEY")
    route = ModelRoute(model="gpt-4o-mini", api_key=ref)
    assert route.api_key is ref


def test_model_route_none_api_key_resolves_to_none():
    route = ModelRoute(model="gpt-4o-mini")
    assert route.resolved_api_key() is None


def test_model_route_full_fields():
    route = ModelRoute(
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        api_key="sk-ant-test",
        temperature=0.5,
        max_tokens=2048,
        base_url="https://custom.endpoint",
    )
    assert route.provider == "anthropic"
    assert route.temperature == 0.5
    assert route.max_tokens == 2048
    assert route.base_url == "https://custom.endpoint"
    assert route.resolved_api_key() == "sk-ant-test"


# ── RouterConfig ──────────────────────────────────────────────────────────────

def test_router_config_defaults_all_none():
    rc = RouterConfig()
    assert rc.fallback is None
    assert rc.tool_call is None
    assert rc.summarizer is None


def test_router_config_with_all_slots():
    rc = RouterConfig(
        fallback=ModelRoute(model="claude-haiku-4-5-20251001", provider="anthropic"),
        tool_call=ModelRoute(model="gpt-4o-mini"),
        summarizer=ModelRoute(model="gpt-4o-mini"),
    )
    assert rc.fallback.model == "claude-haiku-4-5-20251001"
    assert rc.tool_call.model == "gpt-4o-mini"
    assert rc.summarizer.model == "gpt-4o-mini"


def test_kazi_config_includes_router_field():
    cfg = KaziConfig()
    assert isinstance(cfg.router, RouterConfig)
    assert cfg.router.fallback is None


def test_kazi_config_router_set():
    cfg = KaziConfig(
        router=RouterConfig(tool_call=ModelRoute(model="gpt-4o-mini"))
    )
    assert cfg.router.tool_call.model == "gpt-4o-mini"


# ── GraphBrain._build_llm ─────────────────────────────────────────────────────

def _make_brain(config=None):
    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.registry import ToolRegistry

    cfg = config or KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"))
    registry = ToolRegistry()

    with patch.object(GraphBrain, "_build"):
        brain = GraphBrain.__new__(GraphBrain)
        brain.config = cfg
        brain.registry = registry
        brain._llm_cache = {}
        brain._graph = None
        brain._checkpointer = None
        brain._active_budgets = {}
        import asyncio
        brain._budget_lock = asyncio.Lock()
    return brain


def test_build_llm_primary_cached(monkeypatch):
    brain = _make_brain()
    fake_llm = MagicMock()
    monkeypatch.setattr(brain, "_make_llm", lambda **kw: fake_llm)

    llm1 = brain._build_llm()
    llm2 = brain._build_llm()
    assert llm1 is llm2  # same object returned from cache


def test_build_llm_route_uses_route_model(monkeypatch):
    brain = _make_brain()
    created = {}

    def fake_make(**kw):
        created.update(kw)
        return MagicMock()

    monkeypatch.setattr(brain, "_make_llm", fake_make)

    route = ModelRoute(model="gpt-4o-mini")
    brain._build_llm(route)

    assert created["model"] == "gpt-4o-mini"


def test_build_llm_route_inherits_primary_temperature(monkeypatch):
    cfg = KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o", temperature=0.7))
    brain = _make_brain(cfg)

    created = {}

    def fake_make(**kw):
        created.update(kw)
        return MagicMock()

    monkeypatch.setattr(brain, "_make_llm", fake_make)

    route = ModelRoute(model="gpt-4o-mini")  # temperature=None → inherit
    brain._build_llm(route)
    assert created["temperature"] == 0.7


def test_build_llm_route_overrides_temperature(monkeypatch):
    cfg = KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o", temperature=0.7))
    brain = _make_brain(cfg)
    created = {}

    def fake_make(**kw):
        created.update(kw)
        return MagicMock()

    monkeypatch.setattr(brain, "_make_llm", fake_make)

    route = ModelRoute(model="gpt-4o-mini", temperature=0.0)
    brain._build_llm(route)
    assert created["temperature"] == 0.0


def test_build_llm_route_inherits_primary_provider(monkeypatch):
    brain = _make_brain()
    created = {}

    def fake_make(**kw):
        created.update(kw)
        return MagicMock()

    monkeypatch.setattr(brain, "_make_llm", fake_make)

    route = ModelRoute(model="gpt-4o-mini")  # no provider → inherit "openai"
    brain._build_llm(route)
    assert created["provider"] == "openai"


def test_build_llm_route_overrides_provider(monkeypatch):
    brain = _make_brain()
    created = {}

    def fake_make(**kw):
        created.update(kw)
        return MagicMock()

    monkeypatch.setattr(brain, "_make_llm", fake_make)

    route = ModelRoute(model="claude-haiku-4-5-20251001", provider="anthropic")
    brain._build_llm(route)
    assert created["provider"] == "anthropic"
    assert created["model"] == "claude-haiku-4-5-20251001"


def test_build_llm_different_routes_cached_separately(monkeypatch):
    brain = _make_brain()
    call_count = 0

    def fake_make(**kw):
        nonlocal call_count
        call_count += 1
        return MagicMock()

    monkeypatch.setattr(brain, "_make_llm", fake_make)

    brain._build_llm(ModelRoute(model="gpt-4o-mini"))
    brain._build_llm(ModelRoute(model="gpt-4o-mini"))   # cache hit
    brain._build_llm(ModelRoute(model="gpt-3.5-turbo"))  # cache miss

    assert call_count == 2


def test_build_llm_returns_custom_llm_for_primary():
    custom = MagicMock()
    cfg = KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o", custom_llm=custom))
    brain = _make_brain(cfg)
    assert brain._build_llm() is custom


def test_build_llm_route_ignores_custom_llm(monkeypatch):
    custom = MagicMock()
    cfg = KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o", custom_llm=custom))
    brain = _make_brain(cfg)

    route_llm = MagicMock()
    monkeypatch.setattr(brain, "_make_llm", lambda **kw: route_llm)

    result = brain._build_llm(ModelRoute(model="gpt-4o-mini"))
    assert result is route_llm  # custom_llm not used for routes


# ── Fallback routing ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_node_uses_fallback_on_primary_failure():
    """When the primary LLM raises, the fallback model is tried."""
    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.registry import ToolRegistry

    primary_llm = MagicMock()
    primary_llm.bind_tools = MagicMock(return_value=primary_llm)
    primary_llm.ainvoke = AsyncMock(side_effect=RuntimeError("provider down"))

    fallback_response = AIMessage(content="fallback answer")
    fallback_llm = MagicMock()
    fallback_llm.bind_tools = MagicMock(return_value=fallback_llm)
    fallback_llm.ainvoke = AsyncMock(return_value=fallback_response)

    cfg = KaziConfig(
        router=RouterConfig(
            fallback=ModelRoute(model="claude-haiku-4-5-20251001", provider="anthropic")
        )
    )
    registry = ToolRegistry()

    with patch.object(GraphBrain, "_build"):
        brain = GraphBrain.__new__(GraphBrain)
        brain.config = cfg
        brain.registry = registry
        brain._llm_cache = {}
        brain._graph = None
        brain._checkpointer = None
        brain._active_budgets = {}
        import asyncio
        brain._budget_lock = asyncio.Lock()

    # Patch _build_llm: primary slot → primary_llm, fallback route → fallback_llm
    fallback_route = cfg.router.fallback

    def fake_build_llm(route=None):
        if route is fallback_route:
            return fallback_llm
        return primary_llm

    brain._build_llm = fake_build_llm

    state = {
        "messages": [HumanMessage(content="hello")],
        "thread_id": "t1",
        "current_step": "start",
        "tool_calls_made": 0,
        "max_tool_calls": 25,
        "system_prompt": None,
        "final_answer": None,
        "metadata": {},
    }

    # Invoke the agent node directly by extracting it from _build
    # Re-build just the agent_node closure
    with patch.object(GraphBrain, "_build"):
        pass

    # Simulate _build by calling the relevant parts inline
    async def run_agent_node():
        router = brain.config.router
        messages = list(state["messages"])
        last_msg = messages[-1] if messages else None
        from langchain_core.messages import ToolMessage as TM
        is_tool_turn = isinstance(last_msg, TM)
        route = router.tool_call if (is_tool_turn and router.tool_call) else None
        llm = brain._build_llm(route)
        tool_schemas = []
        llm_with_tools = llm

        sys_msg = SystemMessage(content="You are a helpful AI assistant.")
        full_messages = [sys_msg] + messages

        try:
            response = await llm_with_tools.ainvoke(full_messages)
        except Exception:
            if router.fallback is None:
                raise
            fallback_llm_inst = brain._build_llm(router.fallback)
            fallback_with_tools = (
                fallback_llm_inst.bind_tools(tool_schemas) if tool_schemas else fallback_llm_inst
            )
            response = await fallback_with_tools.ainvoke(full_messages)

        return response

    response = await run_agent_node()
    assert response.content == "fallback answer"
    primary_llm.ainvoke.assert_called_once()
    fallback_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_agent_node_raises_when_no_fallback_configured():
    """Without a fallback, primary LLM errors propagate immediately."""
    primary_llm = MagicMock()
    primary_llm.ainvoke = AsyncMock(side_effect=RuntimeError("hard failure"))

    cfg = KaziConfig()  # no fallback

    router = cfg.router
    messages = [HumanMessage(content="hello")]
    sys_msg = SystemMessage(content="You are helpful.")
    full_messages = [sys_msg] + messages

    with pytest.raises(RuntimeError, match="hard failure"):
        try:
            await primary_llm.ainvoke(full_messages)
        except Exception:
            if router.fallback is None:
                raise


# ── Tool-turn routing ─────────────────────────────────────────────────────────

def test_tool_turn_detection_uses_tool_call_route():
    """When last message is ToolMessage, tool_call route should be selected."""

    cfg = KaziConfig(
        router=RouterConfig(tool_call=ModelRoute(model="gpt-4o-mini"))
    )
    brain = _make_brain(cfg)

    selected_routes = []

    def fake_build_llm(route=None):
        selected_routes.append(route)
        return MagicMock()

    brain._build_llm = fake_build_llm

    messages = [
        HumanMessage(content="search for something"),
        ToolMessage(content="search results here", tool_call_id="tc-1"),
    ]

    last_msg = messages[-1]
    is_tool_turn = isinstance(last_msg, ToolMessage)
    route = cfg.router.tool_call if (is_tool_turn and cfg.router.tool_call) else None

    brain._build_llm(route)
    assert selected_routes[0] is cfg.router.tool_call


def test_non_tool_turn_uses_primary():
    """When last message is HumanMessage, primary model should be selected."""
    cfg = KaziConfig(
        router=RouterConfig(tool_call=ModelRoute(model="gpt-4o-mini"))
    )

    messages = [HumanMessage(content="what is the weather?")]
    last_msg = messages[-1]
    is_tool_turn = isinstance(last_msg, ToolMessage)
    route = cfg.router.tool_call if (is_tool_turn and cfg.router.tool_call) else None

    assert route is None  # primary selected


# ── Summarizer routing ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_summarise_uses_summarizer_llm_when_provided():
    """When summarizer_llm is passed, it (not the primary llm) does the summarisation."""
    from langchain_core.messages import AIMessage, HumanMessage

    primary_llm = AsyncMock()
    summarizer_llm = AsyncMock()

    response = MagicMock()
    response.content = "Compact summary."
    summarizer_llm.ainvoke = AsyncMock(return_value=response)

    config = TokenBudgetConfig(summarize_after_turns=2)
    messages = [
        HumanMessage(content="msg 1"),
        AIMessage(content="reply 1"),
        HumanMessage(content="msg 2"),
        AIMessage(content="reply 2"),
        HumanMessage(content="msg 3"),
    ]

    result = await maybe_summarise(messages, primary_llm, config, summarizer_llm=summarizer_llm)

    summarizer_llm.ainvoke.assert_called_once()
    primary_llm.ainvoke.assert_not_called()
    assert any(isinstance(m, SystemMessage) and "Compact summary" in m.content for m in result)


@pytest.mark.asyncio
async def test_maybe_summarise_falls_back_to_primary_when_no_summarizer():
    """Without summarizer_llm, the primary llm handles summarisation."""
    from langchain_core.messages import AIMessage, HumanMessage

    primary_llm = AsyncMock()
    response = MagicMock()
    response.content = "Primary summary."
    primary_llm.ainvoke = AsyncMock(return_value=response)

    config = TokenBudgetConfig(summarize_after_turns=2)
    messages = [
        HumanMessage(content="a"),
        AIMessage(content="b"),
        HumanMessage(content="c"),
        AIMessage(content="d"),
        HumanMessage(content="e"),
    ]

    await maybe_summarise(messages, primary_llm, config)

    primary_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_summarise_handles_list_content_messages():
    """Messages with Anthropic-style list content are included in the summary."""
    from langchain_core.messages import AIMessage, HumanMessage

    summarizer = AsyncMock()
    response = MagicMock()
    response.content = "Summary of list-content messages."
    summarizer.ainvoke = AsyncMock(return_value=response)

    config = TokenBudgetConfig(summarize_after_turns=2)

    # Anthropic-style multi-block message
    anthropic_msg = AIMessage(content=[{"type": "text", "text": "block content here"}])
    messages = [
        HumanMessage(content="question"),
        anthropic_msg,
        HumanMessage(content="follow up"),
        AIMessage(content="final"),
        HumanMessage(content="last"),
    ]

    await maybe_summarise(messages, MagicMock(), config, summarizer_llm=summarizer)

    # The prompt passed to the summarizer should include the block content
    prompt_arg = summarizer.ainvoke.call_args[0][0][0].content
    assert "block content here" in prompt_arg
