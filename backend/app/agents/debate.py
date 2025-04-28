from __future__ import annotations
import os
import textwrap
from typing import List, Dict, Any, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.agents import AgentNode, ConfidenceVote
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from pydantic import BaseModel, Field

# ───────────────────────────────────────────────────────────────
# ❶ CONFIG
PERSONAE = [
    "Macro Strategist",
    "Micro-Factor Analyst",
    "Sentiment Sentinel",
    "Devil’s-Advocate",
    "Risk-Guard",
]

class DebateOutput(BaseModel):
    answer: str
    rationale: str
    confidence: float = Field(ge=0, le=1)

class MessagesState(TypedDict):
    question: str
    context: str
    messages: List[str]       # running log of prior rationales (optional)

# ───────────────────────────────────────────────────────────────
# ❷ LLM factory
def _make_llm():
    """Return GPT-4o-mini if OPENAI_API_KEY is set, else local Phi-3 Mini."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return LlamaCpp(
        model_path="/models/phi3-mini.Q4.gguf",
        n_ctx=4096,
        temperature=0.2,
    )

# ───────────────────────────────────────────────────────────────
# ❸ Persona node builder
def _persona_node(llm, persona: str) -> AgentNode:
    prompt = textwrap.dedent(
        f"""
        <<SYSTEM>>
        You are {persona}.
        1. **Skeleton-of-Thought:** jot 3-5 bullet points outlining the reasoning path.  
        2. **Chain-of-Thought:** expand each bullet into full sentences.  
        Return JSON: {{"answer": "...", "rationale": "...", "confidence": 0-1}}
        <<QUESTION>> {{question}}
        <<CONTEXT>>  {{context}}
        """
    )
    return AgentNode.from_prompt(llm, prompt, DebateOutput)

# ───────────────────────────────────────────────────────────────
# ❹ Build the LangGraph topology
def build_debate_graph(n_agents: int = 5) -> StateGraph:
    llm = _make_llm()
    builder = StateGraph(MessagesState)

    # analyst nodes
    for persona in PERSONAE[:n_agents]:
        builder.add_node(persona, _persona_node(llm, persona))
        builder.add_edge("START", persona)

    # confidence-weighted vote aggregator
    vote = ConfidenceVote()
    builder.add_node("vote", vote)

    # route all analysts → vote → END
    for persona in PERSONAE[:n_agents]:
        builder.add_edge(persona, "vote")
    builder.add_edge("vote", END)

    return builder.compile()

# ───────────────────────────────────────────────────────────────
# ❺ Debate runner with optional early-stop
def run_debate(
    question: str,
    context: str = "",
    max_rounds: int = 3,
    threshold: float = 0.85,
    n_agents: int = 5,
) -> DebateOutput:
    """Run up to *max_rounds* until winning confidence ≥ threshold."""
    graph = build_debate_graph(n_agents)
    state: Dict[str, Any] = {"question": question, "context": context, "messages": []}
    consensus: DebateOutput | None = None

    for r in range(max_rounds):
        # ── one debate pass ─────────────────────────────────────────
        result: DebateOutput = graph.invoke(state)  # type: ignore
        consensus = result
        print(f"» Round {r+1}: {result.answer} (conf={result.confidence:.2f})")

        # ── early exit if high confidence ──────────────────────────
        if result.confidence >= threshold:
            break

        # ── feed rationale back into context for cross-exam ───────
        state["context"] += (
            f"\n\n── Previous round rationale (consensus) ──\n{result.rationale}"
        )
        state["messages"].append(result.model_dump_json())

    # return final consensus (JSON-validated object)
    return consensus  # type: ignore

# ───────────────────────────────────────────────────────────────
# ❻ Quick smoke-test
if __name__ == "__main__":
    out = run_debate(
        question="Is AAPL currently overvalued?",
        context="Daily price chart, P/E 31, forward EPS growth 8 %, Fed stance neutral.",
    )
    print("\nFinal decision →", out.json(indent=2))
