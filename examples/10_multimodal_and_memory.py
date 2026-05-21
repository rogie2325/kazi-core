"""
Example 10: Multi-modal input + long-term user memory + experiment tracking

Covers three features in one example because they compose naturally:
  1. Images in kazi.run() for vision tasks (charts, screenshots, documents)
  2. UserProfile for cross-session user preferences (injected automatically)
  3. ExperimentTracker for MLflow/W&B metric logging

Multi-modal requires a vision-capable model:
  - Claude Sonnet 4.6 / Opus 4.7 (Anthropic)
  - GPT-4o (OpenAI)

Install::

    pip install kazi-core[anthropic]
    pip install mlflow       # optional — for experiment tracking
    pip install wandb        # optional — for W&B tracking
"""
import asyncio
import time

from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider, UserProfile, ExperimentTracker


# ── Multi-modal: analyze images ────────────────────────────────────────────────

async def analyze_chart_from_url():
    """Pass an image URL — the model describes or analyzes it."""
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6")
    )

    async with await Kazi.create(config) as kazi:
        reply = await kazi.run(
            "Describe the key trends in this chart. What's the main takeaway?",
            images=["https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Chocolate-Chip-Cookies-Recipe.jpg/1024px-Banana-Chocolate-Chip-Cookies-Recipe.jpg"],
        )
        print(f"Chart analysis: {reply}")


async def analyze_local_image():
    """Pass a local file path — kazi base64-encodes it automatically."""
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
    )

    async with await Kazi.create(config) as kazi:
        # Multiple images in one turn
        reply = await kazi.run(
            "Compare these two screenshots and describe the UI differences.",
            images=[
                "./screenshots/before.png",
                "./screenshots/after.png",
            ],
        )
        print(f"UI comparison: {reply}")


# ── Long-term user memory ──────────────────────────────────────────────────────

async def user_profile_example():
    """
    Store user facts once. They're injected automatically on every future run().
    Works across sessions, restarts, and modalities (voice + chat share the same profile).
    """
    profile = UserProfile(storage_dir=".kazi_profiles")

    # Store facts about the user (do this at onboarding or after learning them)
    profile.update("alice", {
        "role": "senior data scientist",
        "prefers": "concise bullet-point answers",
        "timezone": "UTC-5",
        "expertise": "Python, SQL, machine learning",
        "company": "Acme Corp",
    })

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6")
    )

    async with await Kazi.create(config) as kazi:
        # user_id= injects the profile preamble into every system prompt
        reply = await kazi.run(
            "Recommend the best approach for feature engineering on tabular data.",
            user_id="alice",
            thread_id="alice:session-42",
        )
        # The model knows Alice is a senior data scientist who prefers bullet points
        print(f"Personalized reply: {reply}")

        # Update profile based on what you learn during the session
        profile.update("alice", {"current_project": "churn prediction model"})

        # Next turn — model now knows her current project too
        reply2 = await kazi.run(
            "What evaluation metrics should I focus on?",
            user_id="alice",
            thread_id="alice:session-42",
        )
        print(f"Context-aware reply: {reply2}")


# ── Experiment tracking ────────────────────────────────────────────────────────

async def experiment_tracking_example():
    """
    Log token counts, cost, and latency to MLflow or W&B.
    Useful for prompt engineering, A/B testing models, and cost optimization.
    """
    tracker = ExperimentTracker(
        backend="mlflow",       # or "wandb"
        project="kazi-agent-evals",
        run_name="claude-sonnet-4-6-baseline",
    )

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6")
    )

    messages = [
        "Summarise the key benefits of RAG vs fine-tuning in 3 bullets.",
        "What are the main failure modes of LLM agents?",
        "Explain chain-of-thought prompting to a junior engineer.",
    ]

    async with await Kazi.create(config) as kazi:
        for msg in messages:
            t0 = time.monotonic()
            result = await kazi.run(msg, track_cost=True, thread_id="eval-session")
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Log everything to MLflow / W&B
            tracker.log_run_result(
                message=msg,
                result=result,
                model=config.llm.model,
                latency_ms=elapsed_ms,
                thread_id="eval-session",
            )

            print(f"Q: {msg}")
            print(f"A: {result.reply[:150]}…")
            print(f"   cost=${result.cost.cost_usd:.4f}  latency={elapsed_ms:.0f}ms")
            print()

    tracker.finish()


# ── Per-tenant isolation ───────────────────────────────────────────────────────

async def tenant_isolation_example():
    """
    SaaS use case: different customers see different tool sets.
    No registry mutation — isolation is enforced at the graph level via tenant_id.
    """
    from kazi import web_search_tool, python_sandbox_tool, sql_query_tool

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
        # Map each tenant to the tools they're allowed to use
        tenant_tools={
            "tenant:free": {"web_search"},
            "tenant:pro": {"web_search", "python_sandbox"},
            "tenant:enterprise": {"web_search", "python_sandbox", "query_company_db"},
        },
    )

    async with await Kazi.create(config) as kazi:
        kazi.registry.register(web_search_tool(), category="search")
        kazi.registry.register(python_sandbox_tool(), category="compute")
        # kazi.registry.register(sql_query_tool("postgresql://..."), name="query_company_db")

        # Free tier — only sees web_search
        reply_free = await kazi.run(
            "Search for Python best practices",
            tenant_id="tenant:free",
            thread_id="free-user-1",
        )

        # Pro tier — sees web_search + python_sandbox
        reply_pro = await kazi.run(
            "Search for datasets and write code to analyse them",
            tenant_id="tenant:pro",
            thread_id="pro-user-1",
        )

        print(f"Free tier: {reply_free[:100]}…")
        print(f"Pro tier: {reply_pro[:100]}…")


if __name__ == "__main__":
    asyncio.run(user_profile_example())
    # asyncio.run(analyze_chart_from_url())
    # asyncio.run(experiment_tracking_example())
    # asyncio.run(tenant_isolation_example())
