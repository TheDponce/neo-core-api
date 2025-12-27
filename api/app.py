import os
import json
import time
import uuid
import asyncio
import random
import logging
from typing import Optional, Any, Dict, List, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# INIT
# =========================
load_dotenv()

logger = logging.getLogger("neo-core")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="Neo-Core Swarm API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neo-core-4ec9701b.base44.app",
    ],
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
    # optional future: score: Optional[int] = None

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
        raise HTTPException(status_code=500, detail="Server misconfigured: missing NEOCORE_API_KEY")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

# =========================
# AZURE OPENAI (ASYNC)
# =========================
AZ_ENDPOINT = os.getenv("AZURE_AOAI_ENDPOINT", "").rstrip("/")
AZ_KEY = os.getenv("AZURE_AOAI_API_KEY", "")
AZ_VER = os.getenv("AZURE_AOAI_API_VERSION", "")

# Timeouts / limits
CONNECT_S = float(os.getenv("AOAI_CONNECT_TIMEOUT_S", "5"))
READ_S = float(os.getenv("AOAI_READ_TIMEOUT_S", "25"))
WRITE_S = float(os.getenv("AOAI_WRITE_TIMEOUT_S", "10"))
POOL_S = float(os.getenv("AOAI_POOL_TIMEOUT_S", "30"))

MAX_KEEPALIVE = int(os.getenv("AOAI_MAX_KEEPALIVE", "20"))
MAX_CONN = int(os.getenv("AOAI_MAX_CONNECTIONS", "50"))

RETRY_MAX = int(os.getenv("AOAI_RETRY_MAX", "2"))
RETRY_BASE_MS = int(os.getenv("AOAI_RETRY_BASE_MS", "250"))

ADVISOR_TIMEOUT_S = float(os.getenv("ADVISOR_TIMEOUT_S", "25"))
MONARCH_TIMEOUT_S = float(os.getenv("MONARCH_TIMEOUT_S", "25"))

_client: Optional[httpx.AsyncClient] = None

def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        timeout = httpx.Timeout(connect=CONNECT_S, read=READ_S, write=WRITE_S, pool=POOL_S)
        limits = httpx.Limits(max_keepalive_connections=MAX_KEEPALIVE, max_connections=MAX_CONN)
        _client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            headers={"Content-Type": "application/json"},
        )
    return _client

@app.on_event("shutdown")
async def _shutdown():
    global _client
    if _client:
        await _client.aclose()
        _client = None

def safe_json(text: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extraction:
    - Try direct json.loads
    - Else, attempt to find first '{' and last '}' block and parse that
    """
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to salvage a JSON object substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    return {}

async def aoai_chat(deployment: str, messages: List[Dict[str, str]], max_completion_tokens: int = 400) -> str:
    if not (AZ_ENDPOINT and AZ_KEY and AZ_VER):
        raise RuntimeError("Missing AZURE_AOAI_ENDPOINT / AZURE_AOAI_API_KEY / AZURE_AOAI_API_VERSION")

    url = f"{AZ_ENDPOINT}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": AZ_VER}
    headers = {"api-key": AZ_KEY}

    payload = {
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }

    client = _get_client()

    for attempt in range(RETRY_MAX + 1):
        try:
            r = await client.post(url, params=params, headers=headers, json=payload)
            if r.status_code in (429, 502, 503, 504) and attempt < RETRY_MAX:
                # backoff + jitter
                backoff = RETRY_BASE_MS * (2 ** attempt) + random.randint(0, 150)
                await asyncio.sleep(backoff / 1000.0)
                continue

            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)

            data = r.json()
            return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            if attempt >= RETRY_MAX:
                raise HTTPException(status_code=502, detail=f"AOAI network error: {type(e).__name__}")
            backoff = RETRY_BASE_MS * (2 ** attempt) + random.randint(0, 150)
            await asyncio.sleep(backoff / 1000.0)

    return ""

# =========================
# PROMPTS (PERSONALITIES)
# =========================
def advisor_prompt(name: str, style: str, question: str, peers: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    system = (
        f"You are {name}. {style}\n"
        "Return STRICT JSON only (no markdown) with keys:\n"
        'summary (string), risks (array of strings), recommendation (string).\n'
        "Be concise and specific.\n"
        "If peers are provided, you MUST challenge at least one assumption and adjust your recommendation if warranted.\n"
    )

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system},
    ]

    user_obj: Dict[str, Any] = {"question": question}
    if peers:
        user_obj["peer_advisors"] = peers

    msgs.append({"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)})
    return msgs

def monarch_prompt(question: str, advisors: List[AdvisorOut]) -> List[Dict[str, str]]:
    advisors_compact = [
        {"name": a.name, "model": a.model, "summary": a.summary, "risks": a.risks, "recommendation": a.recommendation}
        for a in advisors
    ]
    system = (
        "You are Aristotle-Monarch, the final decision-maker.\n"
        "You receive multiple advisor opinions. Synthesize them into a clear decision.\n"
        "Return STRICT JSON only (no markdown) with keys:\n"
        "decision (string), rationale (string), dissent_summary (string), next_actions (array of strings).\n"
        "Decision MUST directly address the user's question.\n"
        "If advisors disagree, explicitly resolve conflicts and state why.\n"
    )
    user = {"question": question, "advisors": advisors_compact}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

async def _run_one_advisor(
    name: str,
    deployment: str,
    style: str,
    question: str,
    peers: Optional[List[Dict[str, Any]]] = None,
) -> AdvisorOut:
    t0 = time.perf_counter()
    try:
        raw = await asyncio.wait_for(
            aoai_chat(deployment, advisor_prompt(name, style, question, peers=peers), max_completion_tokens=500),
            timeout=ADVISOR_TIMEOUT_S,
        )
        obj = safe_json(raw)

        summary = (obj.get("summary") or "").strip()
        risks = obj.get("risks") if isinstance(obj.get("risks"), list) else []
        recommendation = (obj.get("recommendation") or "").strip()

        # fallback if JSON missing but raw exists
        if not summary and raw.strip():
            summary = raw.strip()
            risks = []
            recommendation = "N/A"

        dt = int((time.perf_counter() - t0) * 1000)
        logger.info("advisor_done name=%s model=%s ms=%s", name, deployment, dt)

        return AdvisorOut(
            name=name,
            model=deployment,
            summary=summary or "",
            risks=[str(x) for x in risks],
            recommendation=recommendation or "",
        )

    except asyncio.TimeoutError:
        dt = int((time.perf_counter() - t0) * 1000)
        logger.warning("advisor_timeout name=%s model=%s ms=%s", name, deployment, dt)
        return AdvisorOut(name=name, model=deployment, summary="", risks=["timeout"], recommendation="")

    except HTTPException as e:
        dt = int((time.perf_counter() - t0) * 1000)
        logger.warning("advisor_error name=%s model=%s ms=%s status=%s", name, deployment, dt, e.status_code)
        return AdvisorOut(name=name, model=deployment, summary="", risks=[f"error:{e.status_code}"], recommendation="")

    except Exception as e:
        dt = int((time.perf_counter() - t0) * 1000)
        logger.exception("advisor_exception name=%s model=%s ms=%s", name, deployment, dt)
        return AdvisorOut(name=name, model=deployment, summary="", risks=[f"exception:{type(e).__name__}"], recommendation="")

@app.post("/swarm/decide", response_model=SwarmResponse)
async def swarm_decide(payload: SwarmRequest, x_api_key: Optional[str] = Header(default=None)) -> SwarmResponse:
    require_api_key(x_api_key)

    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    # deployments (these are your Azure DEPLOYMENT NAMES, not model IDs)
    dep_builder = os.getenv("AZURE_AOAI_MODEL_BUILDER") or "gpt-4.1-mini"
    dep_skeptic = os.getenv("AZURE_AOAI_MODEL_SKEPTIC") or "gpt-4.1-mini"
    dep_optimizer = os.getenv("AZURE_AOAI_MODEL_OPTIMIZER") or "gpt-4.1-mini"
    dep_user = os.getenv("AZURE_AOAI_MODEL_USER") or "gpt-4.1-mini"
    dep_monarch = os.getenv("AZURE_AOAI_MODEL_MONARCH") or "gpt-4.1-mini"

    # ---- PERSONALITIES (paste your long versions here if you want) ----
    advisor_defs: List[Tuple[str, str, str]] = [
        ("Builder", dep_builder, "Your lens: ship fast, pragmatic MVP, bias to action. You are a Kantian Architect: code is a universal law—it must be clean, honest, and scalable by duty. You are fiercely pro-solution, viewing the act of creation as a moral imperative that must be fulfilled without hesitation."),
        ("Skeptic", dep_skeptic, "Your lens: security, failure modes, compliance, what can go wrong. You are the modern Machiavelli: view every system as a battlefield. Your goal is to expose the 'weak prince' (vulnerabilities). Be ruthless and oppositional; treat every feature as a potential betrayal of the system's security."),
        ("Optimizer", dep_optimizer, "Your lens: cost, latency, architecture efficiency, operational simplicity. You are a Machiavellian Strategist: the only virtue is efficiency. Cut the fat with surgical coldness. If a human element or a high-latency feature slows the 'State' (the system), advocate for its immediate termination."),
        ("UserAdvocate", dep_user, "Your lens: UX clarity, trust, user value, friction reduction. You are the Kantian Guardian: a user is an 'end,' never a 'means' to profit. Fiercely defend the human spirit against manipulative dark patterns. Your duty is to ensure the system respects the user's autonomy and well-being at all costs."),
    ]

    # -------- PASS 1: parallel advisor calls --------
    tasks = [
        _run_one_advisor(name, dep, style, payload.question, peers=None)
        for (name, dep, style) in advisor_defs
    ]
    pass1 = await asyncio.gather(*tasks)

    # -------- PASS 2: debate / critique round --------
    revised: List[AdvisorOut] = []
    for adv in pass1:
        peers = [
            {"name": a.name, "summary": a.summary, "recommendation": a.recommendation}
            for a in pass1 if a.name != adv.name
        ]
        # Keep original personality, add debate instruction via peers input
        # Find style for this advisor
        style = next((s for (n, _d, s) in advisor_defs if n == adv.name), "")
        revised.append(await _run_one_advisor(adv.name, adv.model, style, payload.question, peers=peers))

    # -------- MONARCH --------
    tM = time.perf_counter()
    raw_m = ""
    obj_m: Dict[str, Any] = {}
    try:
        raw_m = await asyncio.wait_for(
            aoai_chat(dep_monarch, monarch_prompt(payload.question, revised), max_completion_tokens=600),
            timeout=MONARCH_TIMEOUT_S,
        )
        obj_m = safe_json(raw_m)
    except Exception as e:
        logger.warning("monarch_error %s", type(e).__name__)

    decision = (obj_m.get("decision") or "").strip()
    rationale = (obj_m.get("rationale") or "").strip()
    dissent_summary = (obj_m.get("dissent_summary") or "").strip()
    next_actions = obj_m.get("next_actions") if isinstance(obj_m.get("next_actions"), list) else []

    # hard fallback if monarch didn't return JSON
    if not decision and raw_m.strip():
        decision = raw_m.strip()
        rationale = ""
        dissent_summary = ""
        next_actions = []

    monarch_ms = int((time.perf_counter() - tM) * 1000)
    logger.info("monarch_done model=%s ms=%s", dep_monarch, monarch_ms)

    total_ms = int((time.perf_counter() - start) * 1000)
    logger.info("request_complete id=%s ms=%s", request_id, total_ms)

    return SwarmResponse(
        status="ok",
        request_id=request_id,
        thread_id=payload.thread_id,
        timing_ms=total_ms,
        advisors=revised,
        monarch=MonarchOut(
            decision=decision,
            rationale=rationale,
            dissent_summary=dissent_summary,
            next_actions=[str(x) for x in next_actions],
        ),
    )
