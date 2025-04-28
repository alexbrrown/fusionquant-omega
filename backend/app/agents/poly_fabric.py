from __future__ import annotations
import os, textwrap
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.agents import AgentNode, ConfidenceVote
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

# ─── CONFIG ─────────────────────────────────────────────────────────
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
    messages: List[str]

# ─── LLM factory ────────────────────────────────────────────────────
def _make_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─── Persona node builder ───────────────────────────────────────────
def _persona_node(llm, persona: str) -> AgentNode:
    prompt = textwrap.dedent(f"""
        <<SYSTEM>>
        You are {persona}.
        1. **Skeleton-of-Thought:** jot 3–5 bullet points outlining your reasoning.
        2. **Chain-of-Thought:** expand each bullet into full sentences.
        Return JSON: {{"answer":"...","rationale":"...","confidence":0‐1}}
        <<QUESTION>> {{question}}
        <<CONTEXT>>  {{context}}
    """)
    return AgentNode.from_prompt(llm, prompt, DebateOutput)

# ─── Build the debate graph ──────────────────────────────────────────
def build_debate_graph(n_agents: int = 5) -> StateGraph:
    llm = _make_llm()
    builder = StateGraph(MessagesState)
    for persona in PERSONAE[:n_agents]:
        builder.add_node(persona, _persona_node(llm, persona))
        builder.add_edge("START", persona)
    vote = ConfidenceVote()
    builder.add_node("vote", vote)
    for persona in PERSONAE[:n_agents]:
        builder.add_edge(persona, "vote")
    builder.add_edge("vote", END)
    return builder.compile()

# ─── Runner with early-stop ────────────────────────────────────────
def run_debate(
    question: str,
    context: str = "",
    max_rounds: int = 3,
    threshold: float = 0.85,
    n_agents: int = 5,
) -> DebateOutput:
    graph = build_debate_graph(n_agents)
    state: Dict[str, Any] = {"question": question, "context": context, "messages": []}
    consensus = None
    for r in range(max_rounds):
        result = graph.invoke(state)  # type: ignore
        consensus = result
        print(f"» Round {r+1}: {result.answer} (conf={result.confidence:.2f})")
        if result.confidence >= threshold:
            break
        state["context"] += f"\n\n── Prev rationale ──\n{result.rationale}"
        state["messages"].append(result.model_dump_json())
    return consensus  # type: ignore

# ─── CLI smoke-test ────────────────────────────────────────────────
if __name__ == "__main__":
    out = run_debate(
        question="Is AAPL currently overvalued?",
        context="Daily price chart, P/E 31, EPS growth 8 %, Fed neutral.",
    )
    print("\nFinal decision →", out.json(indent=2))
