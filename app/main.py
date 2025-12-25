from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

import os
import time
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

app = FastAPI(title="Neo-Core Swarm API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Response models ----------
class AdvisorOut(BaseModel):
    name: str
    text: str


class MonarchOut(BaseModel):
    decision: str
    rationale: str
    dissent_summary: str
    next_actions: List[str]


class SwarmResponse(BaseModel):
    status: str
    request_id: str
    thread_id: Optional[str]
    timing_ms: int
    advisors: List[AdvisorOut]
    monarch: MonarchOut


# ---------- Request model ----------
class SwarmRequest(BaseModel):
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    question: str
    context: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

    @field_validator("question")
    @classmethod
    def validate_question_length(cls, value: str) -> str:
        if len(value) > 2000:
            raise ValueError("question must be at most 2000 characters")
        return value

    @field_validator("context")
    @classmethod
    def validate_context_length(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and len(value) > 8000:
            raise ValueError("context must be at most 8000 characters")
        return value


# ---------- Errors (structured) ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = str(uuid.uuid4())
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail, "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = str(uuid.uuid4())
    message = exc.errors()[0].get("msg", "Validation error") if exc.errors() else "Validation error"
    return JSONResponse(
        status_code=422,
        content={"status": "error", "error": message, "request_id": request_id},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    message = str(exc) or "Internal server error"
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": message, "request_id": request_id},
    )


# ---------- Auth ----------
def require_api_key(x_api_key: Optional[str]):
    expected = os.getenv("NEOCORE_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing NEOCORE_API_KEY")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/swarm/decide", response_model=SwarmResponse)
def swarm_decide(payload: SwarmRequest, x_api_key: Optional[str] = Header(default=None)) -> SwarmResponse:
    require_api_key(x_api_key)

    request_id = str(uuid.uuid4())
    t0 = time.time()

    advisors = [
        AdvisorOut(name="Builder", text="Position: Build fast MVP. Counter: risk of tech debt."),
        AdvisorOut(name="Skeptic", text="Position: Enforce constraints and security. Counter: may slow delivery."),
        AdvisorOut(name="Optimizer", text="Position: Reduce cost and latency. Counter: premature optimization."),
        AdvisorOut(name="UserAdvocate", text="Position: Keep UX simple and trustworthy. Counter: fewer advanced features."),
    ]

    monarch = MonarchOut(
        decision="Ship MVP with App Service + API key auth first.",
        rationale="Validates Base44 integration and system flow before adding complexity.",
        dissent_summary="Skeptic warns about auth hardening; Optimizer warns about scaling.",
        next_actions=[
            "Deploy API to Azure App Service.",
            "Wire Base44 External Integration.",
            "Replace stub with Azure Foundry calls.",
        ],
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    return SwarmResponse(
        status="ok",
        request_id=request_id,
        thread_id=payload.thread_id,
        timing_ms=elapsed_ms,
        advisors=advisors,
        monarch=monarch,
    )
