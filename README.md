# Neo-Core Swarm API

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in AZURE_* values
uvicorn api.app:app --reload --port 8000
