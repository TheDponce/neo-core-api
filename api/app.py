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

def aoai_chat(deployment: str, messages: List[Dict[str, str]], max_completion_tokens=800) -> str:
    if not (AZ_ENDPOINT and AZ_KEY and AZ_VER):
        raise RuntimeError("Azure OpenAI not configured")

    url = f"{AZ_ENDPOINT}/openai/deployments/{deployment}/chat/completions"
    headers = {"api-key": AZ_KEY, "Content-Type": "application/json"}
    params = {"api-version": AZ_VER}

    payload = {
        "messages": messages,
        "max_completion_tokens": max_completion_tokens
    }

    try:
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=180)
    except requests.exceptions.ReadTimeout:
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=180)

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
You are providing an analytical argument from a specific philosophical or strategic lens.

MODE:
- Strong disagreement is allowed.
- Be direct, rigorous, and precise.
- Critique ideas and assumptions, not people.
- Do not seek consensus.
- Follow all platform content policies.

You are NOT required to return JSON.
Output PLAIN TEXT only.
"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Framework: {style}\n\nQuestion:\n{question}"},
    ]

    if peers:
        messages.append({
            "role": "user",
            "content": f"Other advisor positions:\n{json.dumps(peers)}\n\nRespond by defending your position."
        })

    return messages


def monarch_prompt(question: str, advisors: List[AdvisorOut]):
    system = """
You are Aristotle-Monarch.

You synthesize competing positions into a final decision.

Rules:
- Be concise.
- No preamble.
- No commentary outside JSON.

Return STRICT JSON only:
{
  "decision": string,
  "rationale": string,
  "dissent_summary": string,
  "next_actions": [string]
}

Hard limits:
- decision: 1 sentence
- rationale: max 4 sentences
- dissent_summary: max 3 sentences
- next_actions: 3–6 short items
"""

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps({
                "question": question,
                "advisors": [
                    {"name": a.name, "text": a.text}
                    for a in advisors
                ]
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
    dep_builder = os.getenv("AZURE_AOAI_MODEL_KANTIAN")
    dep_skeptic = os.getenv("AZURE_AOAI_MODEL_MACHIAVELLIAN")
    dep_optimizer = os.getenv("AZURE_AOAI_MODEL_OPTIMIZER")
    dep_user = os.getenv("AZURE_AOAI_MODEL_USER")
    dep_monarch = os.getenv("AZURE_AOAI_MODEL_MONARCH")

    advisor_defs = [
        ("Builder", dep_builder, "Ship fast, pragmatic MVP, bias to action."),
        ("Skeptic", dep_skeptic, "Security, failure modes, adversarial thinking."),
        ("Optimizer", dep_optimizer, "Efficiency, cost, latency, system simplification."),
        ("UserAdvocate", dep_user, "User trust, autonomy, UX clarity."),
    ]

    advisors: List[AdvisorOut] = []

    # ===== PASS 1 =====
    for name, model, style in advisor_defs:
        text = aoai_chat(model, advisor_prompt(name, style, payload.question))
        advisors.append(AdvisorOut(name=name, model=model, text=text))

    # ===== MONARCH =====
    raw_m = aoai_chat(dep_monarch, monarch_prompt(payload.question, advisors), max_completion_tokens=400)
    m = safe_json(raw_m)

    return SwarmResponse(
        status="ok",
        request_id=str(uuid.uuid4()),
        thread_id=payload.thread_id,
        timing_ms=int((time.time() - start) * 1000),
        advisors=advisors,
        monarch=MonarchOut(
            decision=m.get("decision", ""),
            rationale=m.get("rationale", ""),
            dissent_summary=m.get("dissent_summary", ""),
            next_actions=m.get("next_actions", []),
        ),
    )
