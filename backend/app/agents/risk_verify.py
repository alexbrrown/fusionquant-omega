from __future__ import annotations
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from pydantic import BaseModel, Field, ValidationError
import os, json, textwrap, asyncio

class VerifyResult(BaseModel):
    ok: bool
    critiqued_steps: List[str] = Field(default_factory=list)

_PROMPT = textwrap.dedent("""
You are a strict reasoning auditor.
For each numbered step below, output JSON mapping the index to:
  1 → true  if the logic is sound,
  1 → false if you detect hallucination / non-sequitur.

Return **only** JSON like {"1":true,"2":false,...} – no prose.

Steps:
{steps}
""").strip()

# ──────────────────────────────────────────────────────────────────────────────
def _llm():
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0,
                          max_tokens=20)           # deterministic
    # local fallback – change to any compact NLI model you host
    return LlamaCpp(model_path="/models/phi3-mini.Q4.gguf",
                    temperature=0, max_tokens=20)

# ──────────────────────────────────────────────────────────────────────────────
async def _run_async(llm, prompt: str) -> str:
    if hasattr(llm, "ainvoke"):          # ChatOpenAI supports async
        return await llm.ainvoke(prompt)
    # LlamaCpp sync blocking – run in thread executor
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

async def verify_steps_async(steps: List[str]) -> VerifyResult:
    if not steps:
        return VerifyResult(ok=True)

    indexed = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    llm = _llm()
    raw = await _run_async(llm, _PROMPT.format(steps=indexed))

    try:
        verdicts: Dict[str, bool] = json.loads(raw.strip())
    except (json.JSONDecodeError, ValidationError):
        # fall back – conservative: treat as failure
        return VerifyResult(ok=False, critiqued_steps=steps)

    bad = [steps[int(k)-1] for k, v in verdicts.items() if v is False]
    return VerifyResult(ok=not bad, critiqued_steps=bad)

# Synchronous wrapper for existing callers
def verify_steps(steps: List[str]) -> VerifyResult:
    return asyncio.run(verify_steps_async(steps))
