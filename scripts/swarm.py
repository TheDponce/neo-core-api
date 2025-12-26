import json
import os
import subprocess
import sys
import time

OUTFILE = os.getenv("SWARM_OUTFILE", "swarm_last.json")

def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"❌ Missing env var: {name}", flush=True)
        sys.exit(1)
    return v

def write_partial(obj):
    try:
        with open(OUTFILE, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def call_chat(deployment: str, messages, max_completion_tokens=350, timeout_sec=60):
    endpoint = must_env("AZURE_AOAI_ENDPOINT").rstrip("/")
    key = must_env("AZURE_AOAI_API_KEY")
    version = must_env("AZURE_AOAI_API_VERSION")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    payload = {"messages": messages, "max_completion_tokens": max_completion_tokens}

    cmd = [
        "curl", "-sS",
        "--connect-timeout", "10",
        "--max-time", str(timeout_sec),
        "-H", f"api-key: {key}",
        "-H", "Content-Type: application/json",
        url,
        "-d", json.dumps(payload),
    ]

    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    data = json.loads(out)

    if "error" in data:
        raise RuntimeError(json.dumps(data["error"], indent=2))

    return data["choices"][0]["message"]["content"], data

def advisor_prompt(role_name: str, user_prompt: str) -> list:
    system = (
        f"You are the {role_name} advisor in a multi-agent decision swarm. "
        "Return STRICT JSON only with keys: summary, risks, recommendation. "
        "risks must be an array of short strings."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

def monarch_prompt(user_prompt: str, advisor_packets) -> list:
    system = (
        "You are Aristotle-Monarch. You will receive advisor outputs as JSON. "
        "Synthesize them into STRICT JSON only with keys: decision, rationale, dissent_summary, next_actions. "
        "next_actions must be an array of short strings."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "User request:\n" + user_prompt},
        {"role": "user", "content": "Advisor outputs (JSON):\n" + json.dumps(advisor_packets, indent=2)},
    ]

def main():
    load_dotenv()

    user_prompt = " ".join(sys.argv[1:]).strip()
    if not user_prompt:
        print("Usage:\n  python3 scripts/swarm.py \"Your question here\"", flush=True)
        sys.exit(1)

    models = {
        "Kantian": must_env("AZURE_AOAI_MODEL_KANTIAN"),
        "Machiavellian": must_env("AZURE_AOAI_MODEL_MACHIAVELLIAN"),
        "Optimizer": must_env("AZURE_AOAI_MODEL_OPTIMIZER"),
        "User": must_env("AZURE_AOAI_MODEL_USER"),
        "Monarch": must_env("AZURE_AOAI_MODEL_MONARCH"),
    }

    result = {"advisors": [], "monarch": None, "meta": {"started_at": time.time()}}
    write_partial(result)

    advisor_order = ["Kantian", "Machiavellian", "Optimizer", "User"]
    total_calls = len(advisor_order) + 1

    for i, name in enumerate(advisor_order, start=1):
        print(f"[{i}/{total_calls}] Calling {name} ({models[name]})...", flush=True)
        try:
            content, _raw = call_chat(models[name], advisor_prompt(name, user_prompt), max_completion_tokens=300, timeout_sec=90)
            try:
                packet = json.loads(content)
            except Exception:
                packet = {"summary": content, "risks": [], "recommendation": "N/A"}
            packet["name"] = name
            packet["model"] = models[name]
            result["advisors"].append(packet)
            write_partial(result)
            print(f"    ✓ {name} done", flush=True)
        except Exception as e:
            err = str(e)
            result["advisors"].append({"name": name, "model": models[name], "error": err})
            write_partial(result)
            print(f"    ✗ {name} ERROR: {err}", flush=True)

    print(f"[{total_calls}/{total_calls}] Calling Monarch ({models['Monarch']})...", flush=True)
    try:
        content, _raw = call_chat(models["Monarch"], monarch_prompt(user_prompt, result["advisors"]), max_completion_tokens=500, timeout_sec=120)
        try:
            monarch_json = json.loads(content)
        except Exception:
            monarch_json = {"decision": content, "rationale": "", "dissent_summary": "", "next_actions": []}
        result["monarch"] = monarch_json
        result["meta"]["finished_at"] = time.time()
        write_partial(result)
        print("    ✓ Monarch done", flush=True)
    except Exception as e:
        err = str(e)
        result["monarch"] = {"error": err}
        result["meta"]["finished_at"] = time.time()
        write_partial(result)
        print(f"    ✗ Monarch ERROR: {err}", flush=True)

    print(f"\nSaved: {OUTFILE}", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
