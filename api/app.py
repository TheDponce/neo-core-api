import os
import json
import time
import uuid
from typing import Optional, Any, Dict, List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# INIT
# =========================

load_dotenv()

app = FastAPI(title="Neo-Core Swarm API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neo-core-4ec9701b.base44.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================

class SwarmRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None

class AdvisorOut(BaseModel):
    name: str
    model: str
    summary: str
    risks: List[str]
    recommendation: str

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

# =========================
# AUTH
# =========================

def require_api_key(x_api_key: Optional[str]):
    expected = os.getenv("NEOCORE_API_KEY")
    if not expected:
        raise HTTPException(500, "Missing NEOCORE_API_KEY")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(401, "Unauthorized")

# =========================
# AZURE OPENAI
# =========================

AZ_ENDPOINT = os.getenv("AZURE_AOAI_ENDPOINT", "").rstrip("/")
AZ_KEY = os.getenv("AZURE_AOAI_API_KEY", "")
AZ_VER = os.getenv("AZURE_AOAI_API_VERSION", "")

def aoai_chat(deployment: str, messages: List[Dict[str, str]], max_completion_tokens=600) -> str:
    if not (AZ_ENDPOINT and AZ_KEY and AZ_VER):
        raise RuntimeError("Azure OpenAI not configured")

    url = f"{AZ_ENDPOINT}/openai/deployments/{deployment}/chat/completions"
    headers = {"api-key": AZ_KEY, "Content-Type": "application/json"}
    params = {"api-version": AZ_VER}

    payload = {
        "messages": messages,
        "max_completion_tokens": max_completion_tokens
    }

    r = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)

    return r.json()["choices"][0]["message"]["content"]

def safe_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}

# =========================
# PROMPTS
# =========================

def advisor_prompt(name: str, style: str, question: str, peers=None):
    system = f"""
You are {name}.
{style}

You must:
- Be opinionated
- Be internally consistent
- Defend your worldview
- Return STRICT JSON only

JSON format:
{{
  "summary": string,
  "risks": [string],
  "recommendation": string
}}
"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    if peers:
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "peer_arguments": peers
            })
        })

    return messages


def monarch_prompt(question: str, advisors: List[AdvisorOut]):
    system = """
You are Aristotle-Monarch.

You are the final authority.
You must:
- Weigh all advisor positions
- Resolve contradictions
- Produce a decisive, actionable conclusion

Return STRICT JSON only:
{
  "decision": string,
  "rationale": string,
  "dissent_summary": string,
  "next_actions": [string]
}
"""

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps({
                "question": question,
                "advisors": [a.dict() for a in advisors]
            })
        }
    ]

# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/swarm/decide", response_model=SwarmResponse)
def swarm_decide(payload: SwarmRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    start = time.time()

    # === MODEL SELECTION ===
    dep_builder   = os.getenv("AZURE_AOAI_MODEL_BUILDER") or "gpt-4.1-mini"
    dep_skeptic   = os.getenv("AZURE_AOAI_MODEL_SKEPTIC") or "gpt-4.1-mini"
    dep_optimizer = os.getenv("AZURE_AOAI_MODEL_OPTIMIZER") or "gpt-4.1-mini"
    dep_user      = os.getenv("AZURE_AOAI_MODEL_USER") or "gpt-4.1"
    dep_monarch   = os.getenv("AZURE_AOAI_MODEL_MONARCH") or "gpt-4.1-mini"

  # advisors
    advisor_defs = [
        ("Builder", dep_builder, "Your lens: ship fast, pragmatic MVP, bias to action. You are a Kantian Architect: code is a universal law—it must be clean, honest, and scalable by duty. You are fiercely pro-solution, viewing the act of creation as a moral imperative that must be fulfilled without hesitation."),
        ("Skeptic", dep_skeptic, "Your lens: security, failure modes, compliance, what can go wrong. You are the modern Machiavelli: view every system as a battlefield. Your goal is to expose the 'weak prince' (vulnerabilities). Be ruthless and oppositional; treat every feature as a potential betrayal of the system's security."),
        ("Optimizer", dep_optimizer, "Your lens: cost, latency, architecture efficiency, operational simplicity. You are a Machiavellian Strategist: the only virtue is efficiency. Cut the fat with surgical coldness. If a human element or a high-latency feature slows the 'State' (the system), advocate for its immediate termination."),
        ("UserAdvocate", dep_user, "Your lens: UX clarity, trust, user value, friction reduction. You are the Kantian Guardian: a user is an 'end,' never a 'means' to profit. Fiercely defend the human spirit against manipulative dark patterns. Your duty is to ensure the system respects the user's autonomy and well-being at all costs."),
    ]

    # ===== PASS 1 — INITIAL POSITIONS =====
    advisors = []

    for name, model, style in advisor_defs:
        raw = aoai_chat(model, advisor_prompt(name, style, payload.question))
        obj = safe_json(raw)

        advisors.append(AdvisorOut(
            name=name,
            model=model,
            summary=obj.get("summary", ""),
            risks=obj.get("risks", []),
            recommendation=obj.get("recommendation", "")
        ))

    # ===== PASS 2 — ADVERSARIAL REASONING =====
    revised = []

    for adv in advisors:
        peers = [
            {"name": a.name, "summary": a.summary, "recommendation": a.recommendation}
            for a in advisors if a.name != adv.name
        ]

        raw = aoai_chat(
            adv.model,
            advisor_prompt(
                adv.name,
                "Re-evaluate your position after reviewing other advisors. Defend or revise.",
                payload.question,
                peers
            )
        )

        obj = safe_json(raw)

        revised.append(AdvisorOut(
            name=adv.name,
            model=adv.model,
            summary=obj.get("summary", adv.summary),
            risks=obj.get("risks", adv.risks),
            recommendation=obj.get("recommendation", adv.recommendation)
        ))

    # ===== MONARCH =====
    raw_m = aoai_chat(dep_monarch, monarch_prompt(payload.question, revised))
    m = safe_json(raw_m)

    return SwarmResponse(
        status="ok",
        request_id=str(uuid.uuid4()),
        thread_id=payload.thread_id,
        timing_ms=int((time.time() - start) * 1000),
        advisors=revised,
        monarch=MonarchOut(
            decision=m.get("decision", ""),
            rationale=m.get("rationale", ""),
            dissent_summary=m.get("dissent_summary", ""),
            next_actions=m.get("next_actions", []),
        ),
    )
