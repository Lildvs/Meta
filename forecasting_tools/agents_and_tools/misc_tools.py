import asyncio
import os

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.agent_wrappers import AgentTool, agent_tool
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.forecast_helpers.metaculus_api import (
    MetaculusApi,
    MetaculusQuestion,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
# NOTE: Avoid importing orchestrate_research at module load time to prevent circular imports.


@agent_tool
async def get_general_news_with_asknews(topic: str) -> str:
    """
    Get general news context for a topic using AskNews.
    This will provide a list of news articles and their summaries
    """
    # Gracefully degrade if AskNews credentials are not provided.
    try:
        return await AskNewsSearcher().get_formatted_news_async(topic)
    except ValueError as e:
        if "ASKNEWS_CLIENT_ID" in str(e) or "ASKNEWS_SECRET" in str(e):
            # Keys missing – just return an empty string so the caller can
            # continue without failing the entire run.
            return ""
        raise


@agent_tool
async def perplexity_pro_search(query: str) -> list[dict[str, str]]:  # noqa: D401
    """Deep research via Perplexity.

    Returns a list of ``ResearchSnippet``-style dictionaries rather than a long
    block of text so that the orchestrator can merge them with other sources.
    Each snippet has keys:
    • **source** – "perplexity"
    • **text**   – a concise sentence + markdown link

    Because the LiteLLM wrapper exposes citations & search_results via
    ``response.model_extra``, we use ``litellm.acompletion`` directly to get the
    raw payload.
    """

    try:
        from litellm import acompletion  # noqa: WPS433 – optional heavyweight import
    except ImportError:  # pragma: no cover
        # litellm is already an application dependency, but guard anyway
        return []

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return []

    try:
        response = await acompletion(
            model="perplexity/sonar-pro",
            messages=[{"role": "user", "content": query}],
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            extra_headers={"Content-Type": "application/json"},
        )
    except Exception:  # noqa: BLE001
        return []

    snippets: list[dict[str, str]] = []

    # 1) Convert search_results into snippets (preferred – has title)
    search_results = (response.model_extra or {}).get("search_results")  # type: ignore[attr-defined]
    if isinstance(search_results, list):
        for res in search_results:
            title = res.get("title") or "Perplexity result"
            url = res.get("url") or ""
            date = res.get("date")
            date_txt = f" ({date})" if date else ""
            snippets.append({
                "source": "perplexity",
                "text": f"**{title}**{date_txt}\n[link]({url})",
            })

    # 2) Fallback – use bare citations if search_results missing
    if not snippets:
        citations = (response.model_extra or {}).get("citations")  # type: ignore[attr-defined]
        if isinstance(citations, list):
            for url in citations:
                snippets.append({
                    "source": "perplexity",
                    "text": f"Perplexity citation: [link]({url})",
                })

    # 3) Still nothing? create one snippet from the first 200 chars of content
    if not snippets:
        content_preview = response.choices[0].message.content[:200]
        snippets.append({"source": "perplexity", "text": content_preview})

    return snippets


@agent_tool
async def perplexity_quick_search(query: str) -> str:
    """
    Performs a quick search using Perplexity AI.
    Suitable for fast information retrieval and basic fact-checking.
    """
    model = GeneralLlm(
        model="perplexity/sonar",
        temperature=0,
    )
    response = await model.invoke(query)
    return response


@agent_tool
async def smart_searcher_search(query: str) -> str:
    """
    Performs an intelligent web search using SmartSearcher.
    Uses multiple search queries and advanced filtering for comprehensive results.
    """
    return await SmartSearcher(model="gpt-4o-mini").invoke(query)


@agent_tool
def grab_question_details_from_metaculus(
    url_or_id: str | int,
) -> MetaculusQuestion:
    """
    This function grabs the details of a question from a Metaculus URL or ID.
    """
    if isinstance(url_or_id, int):
        question = MetaculusApi.get_question_by_post_id(url_or_id)
    else:
        question = MetaculusApi.get_question_by_url(url_or_id)
    question.api_json = {}
    return question


@agent_tool
def grab_open_questions_from_tournament(
    tournament_id_or_slug: int | str,
) -> list[MetaculusQuestion]:
    """
    This function grabs the details of all questions from a Metaculus tournament.
    """
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        tournament_id_or_slug
    )
    for question in questions:
        question.api_json = {}
    return questions


def create_tool_for_forecasting_bot(
    bot_or_class: type[ForecastBot] | ForecastBot,
) -> AgentTool:
    if isinstance(bot_or_class, type):
        bot = bot_or_class()
    else:
        bot = bot_or_class

    description = clean_indents(
        """
        Forecast a SimpleQuestion (simplified binary, numeric, or multiple choice question) using a forecasting bot.
        """
    )

    @agent_tool(description_override=description)
    def forecast_question_tool(question: SimpleQuestion) -> str:

        metaculus_question = (
            SimpleQuestion.simple_questions_to_metaculus_questions([question])[
                0
            ]
        )
        task = bot.forecast_question(metaculus_question)
        report = asyncio.run(task)
        return report.explanation

    return forecast_question_tool


# ---------------------------------------------------
# Unified Research Tool (uses orchestrator)
# ---------------------------------------------------


@agent_tool
async def run_research(query: str, depth: str = "quick") -> str:  # noqa: D401
    """Run multi-source research.

    • **quick**  – SmartSearcher + AskNews if keys
    • **deep**   – quick sources **plus** Perplexity deep search

    Returns a markdown bullet list of merged snippets with source tags.
    """

    if depth not in {"quick", "deep"}:
        raise ValueError("depth must be 'quick' or 'deep'")

    from forecasting_tools.forecast_helpers.research_orchestrator import orchestrate_research  # noqa: WPS433 – local import to avoid circular dependency

    snippets = await orchestrate_research(query, depth=depth)  # type: ignore[arg-type]

    # Simple markdown formatting for now
    lines: list[str] = ["### Research findings:\n"]
    for snip in snippets:
        src = snip["source"]
        txt = snip["text"].strip()
        if not txt:
            continue
        lines.append(f"* **{src}** – {txt}")

    return "\n".join(lines)
