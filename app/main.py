from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import time
import uuid

app = FastAPI(title="Neo-Core Swarm API", version="0.1.0")

class SwarmRequest(BaseModel):
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    question: str
    context: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok"}

def require_api_key(x_api_key: Optional[str]):

    expected = os.getenv("NEOCORE_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing NEOCORE_API_KEY")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/swarm/decide")
def swarm_decide(payload: SwarmRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    request_id = str(uuid.uuid4())
    t0 = time.time()

    advisors = [
        {"name": "Builder", "text": "Position: Build fast MVP. Counter: risk of tech debt."},
        {"name": "Skeptic", "text": "Position: Enforce constraints and security. Counter: may slow delivery."},
        {"name": "Optimizer", "text": "Position: Reduce cost and latency. Counter: premature optimization."},
        {"name": "UserAdvocate", "text": "Position: Keep UX simple and trustworthy. Counter: fewer advanced features."},
    ]

    monarch = {
        "decision": "Ship MVP with App Service + API key auth first.",
        "rationale": "Validates Base44 integration and system flow before adding complexity.",
        "dissent_summary": "Skeptic warns about auth hardening; Optimizer warns about scaling.",
        "next_actions": [
            "Deploy API to Azure App Service.",
            "Wire Base44 External Integration.",
            "Replace stub with Azure Foundry calls."
        ]
    }

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "request_id": request_id,
        "thread_id": payload.thread_id,
        "timing_ms": elapsed_ms,
        "advisors": advisors,
        "monarch": monarch
    }
