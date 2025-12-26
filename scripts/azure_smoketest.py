import os
from dotenv import load_dotenv
from openai import OpenAI

print("Loading .env...")
load_dotenv()

endpoint = os.getenv("AZURE_AOAI_ENDPOINT")
key = os.getenv("AZURE_AOAI_API_KEY")
version = os.getenv("AZURE_AOAI_API_VERSION")

# Try MONARCH first, but you can swap to ADVISOR to compare
deployment = os.getenv("AZURE_AOAI_MODEL_MONARCH") or os.getenv("AZURE_AOAI_MODEL_ADVISOR")

print("endpoint:", endpoint)
print("deployment:", deployment)
print("key present:", bool(key))
print("api version:", version)

if not all([endpoint, key, deployment, version]):
    print("\n❌ Missing environment variables.")
    exit(1)

client = OpenAI(
    api_key=key,
    base_url=f"{endpoint}/openai/deployments/{deployment}",
    default_query={"api-version": version},
)

print("\nSending test request...")

resp = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "user", "content": "Reply with exactly: Azure connection successful"}
    ],
    max_completion_tokens=200,
)

choice = resp.choices[0]
content = choice.message.content

print("\n--- PARSED ---")
print("finish_reason:", getattr(choice, "finish_reason", None))
print("content repr:", repr(content))

print("\n--- RAW (for debugging) ---")
try:
    # openai-python objects usually support model_dump_json()
    print(resp.model_dump_json(indent=2))
except Exception:
    # fallback
    print(resp)
