# backend/app/api/api_v1/api.py
from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

# 1) built-in utils
from app.api.api_v1.endpoints.utils import router as utils_router

# 2) your new agents
from app.agents.poly_fabric import run_debate
from app.agents.risk_verify import verify_steps

api_router = APIRouter()

# ↳ /api/v1/utils
api_router.include_router(utils_router, prefix="/utils", tags=["utils"])

# ↳ /api/v1/quant
router_quant = APIRouter(prefix="/quant", tags=["quant"])

class DebateReq(BaseModel):
    question: str
    context: str | None = ""

@router_quant.post("/poly-debate")
def poly_debate(req: DebateReq):
    return run_debate(req.question, req.context or "")

class VerifyReq(BaseModel):
    steps: List[str]

@router_quant.post("/verify")
def verify(req: VerifyReq):
    return verify_steps(req.steps)

api_router.include_router(router_quant)
