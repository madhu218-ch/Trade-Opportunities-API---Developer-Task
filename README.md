```markdown
# Trade Opportunities API — Developer Task

A FastAPI service that analyzes market data and returns trade-opportunity markdown reports for specified sectors (e.g., "pharmaceuticals", "technology", "agriculture").

This repository includes:
- Async FastAPI application
- Single endpoint: GET /analyze/{sector} which returns a structured markdown report
- Session management with JWT-based authentication and guest tokens
- Per-session rate limiting (in-memory)
- Basic input validation and security best-practices
- Integration points for Google Gemini (configurable) with a mock fallback
- Data collection via DuckDuckGo search results (in-memory scraping)
- Clear separation between API, data collection, AI analysis, auth, and rate limiter
- No persistent database — in-memory storage only

Quick status: minimal working prototype and fallback logic for environments without a Gemini key. The system handles external API failures gracefully and returns helpful error messages.

---

## Files included
- app/__init__.py — package marker
- app/main.py — FastAPI app and endpoints
- app/collector.py — market data collection (DuckDuckGo)
- app/ai_client.py — Gemini client adapter + mock analyzer
- app/auth.py — JWT-based auth and session tracking
- app/rate_limiter.py — per-session in-memory rate limiting
- app/models.py — Pydantic request/response models
- requirements.txt — Python dependencies
- .env.example — environment variable examples
- README.md — this file

---

## Requirements

- Python 3.10+
- Recommended: create a virtual environment

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment variables

Create a `.env` file (or set environment vars) with values similar to `.env.example`:

- SECRET_KEY — secret used to sign JWT tokens (set a secure random string)
- ACCESS_TOKEN_EXPIRE_MINUTES — token expiry in minutes (e.g., 60)
- GEMINI_API_KEY — (optional) API key for Google Gemini or other LLM
- GEMINI_API_URL — (optional) REST endpoint for the LLM
- RATE_LIMIT_AUTH_PER_MINUTE — (optional) e.g. 60
- RATE_LIMIT_GUEST_PER_MINUTE — (optional) e.g. 5

See `.env.example` for details.

---

## Running the app

Start with uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive docs:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## Authentication & Session Flow

1. Obtain a token for a test user:
```bash
curl -X POST "http://127.0.0.1:8000/auth/token" -H "Content-Type: application/json" \
 -d '{"username":"dev","password":"devpass"}'
```
Response contains `access_token` (JWT). Example test user included: `dev` / `devpass`.

2. Or obtain a guest token (limited rate & privileges):
```bash
curl -X POST "http://127.0.0.1:8000/auth/guest"
```

3. Use the token to call the analyze endpoint:
```bash
curl -H "Authorization: Bearer <ACCESS_TOKEN>" \
  "http://127.0.0.1:8000/analyze/pharmaceuticals"
```

The response content-type is `text/markdown` and returns a structured markdown report you can save as `pharmaceuticals.md`.

---

## Endpoint Details

GET /analyze/{sector}
- Path parameter:
  - sector (string) — required, 1-50 characters, letters, spaces, dashes allowed
- Authentication:
  - Bearer token (JWT) required (supports guest tokens)
- Rate limiting:
  - Enforced per token (defaults: auth=60/min, guest=5/min)
- Response:
  - 200 OK — `text/markdown` body with the market analysis report
  - 401 Unauthorized — auth problems
  - 429 Too Many Requests — rate limit exceeded
  - 503 Service Unavailable — external API or LLM failure

---

## How the system works (high level)

1. API receives a sector name and validates it.
2. Session is authenticated and rate limit checked.
3. Data collection layer queries DuckDuckGo for recent headlines / links relating to the sector (top N results).
4. AI client constructs a prompt using collected data and sends it to Gemini if configured — otherwise it runs a mock summarization & analysis.
5. The AI response is converted into a structured markdown template including:
   - Title & timestamp
   - Summary of current market conditions
   - Top news items (with links)
   - Trade opportunities (long/short ideas)
   - Risk & caveats
   - Suggested watchlist and next steps
   - Sources
6. API returns markdown content (suitable to save as .md).

---

## Notes & Extensibility

- Gemini integration is an adapter — you can wire any LLM by setting GEMINI_API_URL and GEMINI_API_KEY and matching the expected request shape in app/ai_client.py.
- Data collection is modular; you can swap DuckDuckGo with NewsAPI, web scrapers, or other sources.
- All storage is in-memory; to persist usage/session data across restarts replace the in-memory stores with a persistent store.

---

## Example Output (short excerpt)
The returned markdown will look like:

```markdown
# Trade Opportunity Report — Pharmaceuticals
_Across India — generated: 2025-12-23T12:34:56Z_

## Executive Summary
...

## Top News & Data
- [Headline 1](https://...) — source snippet
- [Headline 2](https://...)

## Trade Opportunities
1. Long: PharmaCo — Rationale, catalysts, entries, targets
2. Short: BioX — Rationale, risk, stop-loss

## Sources
- https://...
```

---

## Development / Tests

- The prototype is intended to be implemented and iterated on within 0–1 day.
- Logging is enabled and error conditions return informative JSON or HTTP codes.
- To extend, add additional collection sources, more advanced LLM prompts, or persistent session storage.

---

## Security & Rate-limiting

- JWT tokens are used for sessions. Guest tokens are allowed but rate-limited.
- Input validation on sector param prevents injection attacks.
- Rate limiting prevents abuse (configurable).
- All external calls time out and are handled; transient failures are surfaced as 503.

---

## Next steps I can help with

- Wire a real Google Gemini API call (if you provide API details / willingness).
- Add additional data sources (NewsAPI, Google News scraping, exchanges).
- Add unit tests and CI (GitHub Actions) to run linting and fast tests.
- Deploy to Cloud Run, ECS, or a small VM.

---

If you'd like, I can now:
1) Generate the full code files (FastAPI app + modules) and a runnable requirements.txt (I included them as files in this repo), or
2) Customize rate limits, provide a production-ready Gemini call example, or
3) Produce deploy instructions (Dockerfile + GitHub Actions workflow).

Which would you like next?
```
```


```python name=app/__init__.py
# Package marker for the app package.
```

```python name=app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class TokenRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class NewsItem(BaseModel):
    title: str
    snippet: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None

class AnalysisInput(BaseModel):
    sector: str
    news: List[NewsItem] = Field(default_factory=list)
```

```python name=app/auth.py
import time
from typing import Optional, Dict
import os
from jose import jwt, JWTError
from fastapi import HTTPException, status, Header
from pydantic import BaseModel
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_change_me")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# In-memory "user" store (for demo)
USERS = {
    "dev": {"username": "dev", "password": "devpass", "role": "developer"},
}

# In-memory session store: token -> session info
SESSIONS: Dict[str, Dict] = {}

class TokenData(BaseModel):
    username: Optional[str] = None
    scope: Optional[str] = "auth"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    # register session
    SESSIONS[token] = {
        "username": data.get("sub"),
        "created_at": int(time.time()),
        "expires_at": int(expire.timestamp()),
        "role": data.get("role", "guest"),
        "requests": 0,
    }
    return token

def decode_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return TokenData(username=username, scope=payload.get("scope", "auth"))
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

async def get_current_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    parts = authorization.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth header format")
    token = parts[1]
    # Validate jwt signature & expiry
    decode_token(token)
    # Ensure session exists and not expired
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session not found or expired")
    if session["expires_at"] < int(time.time()):
        # cleanup
        del SESSIONS[token]
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return token

# Utility functions
def authenticate_user(username: str, password: str):
    user = USERS.get(username)
    if not user or user["password"] != password:
        return None
    return {"username": username, "role": user.get("role", "developer")}
```

```python name=app/rate_limiter.py
import time
from fastapi import HTTPException, status
import os

# Simple in-memory sliding window rate limiter per token
# token -> list of request timestamps (seconds)
REQUEST_LOG = {}

AUTH_LIMIT = int(os.getenv("RATE_LIMIT_AUTH_PER_MINUTE", 60))
GUEST_LIMIT = int(os.getenv("RATE_LIMIT_GUEST_PER_MINUTE", 5))
WINDOW = 60  # seconds

def is_guest_token(token: str, sessions):
    sess = sessions.get(token)
    if not sess:
        return True
    return sess.get("role", "guest") == "guest"

def check_rate_limit(token: str, sessions):
    now = int(time.time())
    window_start = now - WINDOW
    entries = REQUEST_LOG.setdefault(token, [])
    # drop old
    while entries and entries[0] <= window_start:
        entries.pop(0)
    # determine limit
    limit = GUEST_LIMIT if is_guest_token(token, sessions) else AUTH_LIMIT
    if len(entries) >= limit:
        # too many requests in window
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"Rate limit exceeded ({limit} per {WINDOW}s)")
    # allow and add timestamp
    entries.append(now)
    # update session counter if present
    sess = sessions.get(token)
    if sess is not None:
        sess["requests"] = sess.get("requests", 0) + 1
```

```python name=app/collector.py
import asyncio
from typing import List
from duckduckgo_search import ddg
from .models import NewsItem
import logging

logger = logging.getLogger("collector")

# Wrapping ddg (sync) into async using run_in_executor
async def collect_market_data(sector: str, max_results: int = 6) -> List[NewsItem]:
    loop = asyncio.get_running_loop()
    try:
        # ddg returns list of dicts with keys: title, href, body
        results = await loop.run_in_executor(None, lambda: ddg(f"{sector} India news", max_results=max_results))
        items = []
        if not results:
            return items
        for r in results:
            items.append(NewsItem(title=r.get("title") or "No title",
                                  snippet=r.get("body"),
                                  url=r.get("href"),
                                  source=r.get("href")))
        return items
    except Exception as e:
        logger.exception("Error collecting market data")
        return []
```

```python name=app/ai_client.py
import os
import httpx
import json
from typing import List
from datetime import datetime
from .models import NewsItem
import logging

logger = logging.getLogger("ai_client")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

# Timeout for external LLM calls
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20.0"))

def build_prompt(sector: str, news: List[NewsItem]) -> str:
    timestamp = datetime.utcnow().isoformat() + "Z"
    prompt = [
        f"You are an expert market analyst. Provide a structured markdown market analysis report for the sector: '{sector}' (India).",
        f"Date (UTC): {timestamp}",
        "Include: Executive Summary, Top News bullet list, Trade Opportunities (actionable long/short ideas), Risk & Caveats, Suggested Watchlist, and Sources.",
        "Use the provided news items below verbatim as the source material. Cite sources as links.",
        "",
        "=== SOURCE NEWS ITEMS ===",
    ]
    for i, n in enumerate(news, start=1):
        prompt.append(f"{i}. Title: {n.title}\n   Snippet: {n.snippet or ''}\n   URL: {n.url or ''}\n")
    prompt.append("\n=== END OF SOURCES ===\n")
    prompt.append("Produce the report in markdown only. Do not include any extra commentary about availability or API information.")
    return "\n".join(prompt)

async def analyze_with_gemini(sector: str, news: List[NewsItem]) -> str:
    """
    Try to call configured Gemini (or other LLM). If not configured or call fails,
    fallback to local mock_analysis.
    """
    prompt = build_prompt(sector, news)
    # If Gemini configured, try it
    if GEMINI_API_KEY and GEMINI_API_URL:
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "max_output_tokens": 1024}
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                r = await client.post(GEMINI_API_URL, headers=headers, json=payload)
                if r.status_code != 200:
                    logger.error("LLM error status: %s body: %s", r.status_code, r.text)
                    return mock_analysis(sector, news)
                data = r.json()
                # Attempt to find textual output in various fields
                text = None
                if isinstance(data, dict):
                    # common shapes: {'output': '...'} or {'choices': [{'text': '...'}]}
                    text = data.get("output") or data.get("text")
                    if not text and "choices" in data:
                        try:
                            text = data["choices"][0].get("text") or data["choices"][0].get("output")
                        except Exception:
                            pass
                if not text:
                    # if it's nested in 'candidates' or other fields:
                    text = json.dumps(data)
                return text
        except Exception:
            logger.exception("Error calling LLM, falling back to mock analyzer")
            return mock_analysis(sector, news)
    else:
        # No Gemini configured -> mock
        return mock_analysis(sector, news)

def mock_analysis(sector: str, news: List[NewsItem]) -> str:
    """
    Lightweight heuristic-based markdown generator for demo / offline mode.
    """
    now = datetime.utcnow().isoformat() + "Z"
    lines = []
    lines.append(f"# Trade Opportunity Report — {sector.title()}")
    lines.append(f"_Generated (UTC): {now}_\n")
    # Executive summary: simple aggregation
    lines.append("## Executive Summary\n")
    if news:
        lines.append(f"Found {len(news)} recent items related to the {sector} sector in India. Below are summarised insights and suggested trade ideas based on news flow and observable signals.\n")
    else:
        lines.append("No significant news items found. Use caution: limited data.\n")
    # Top news
    lines.append("## Top News & Data\n")
    if news:
        for n in news[:6]:
            t = n.title or "No title"
            s = (n.snippet[:240] + "...") if n.snippet and len(n.snippet) > 240 else (n.snippet or "")
            lines.append(f"- [{t}]({n.url or ''}) — {s}")
    else:
        lines.append("- No news items collected.\n")
    # Trade opportunities (heuristic)
    lines.append("\n## Trade Opportunities\n")
    if news:
        # naive heuristics: if words like 'approval' 'booster' -> bullish opportunity example
        bullish = []
        bearish = []
        for n in news:
            txt = (n.title or "") + " " + (n.snippet or "")
            low = txt.lower()
            if any(w in low for w in ["approval", "positive", "beats", "gain", "growth", "acquisition", "win"]):
                bullish.append(n)
            if any(w in low for w in ["loss", "decline", "fall", "recall", "warning", "regulatory", "probe", "scandal"]):
                bearish.append(n)
        idx = 1
        if bullish:
            for b in bullish[:3]:
                lines.append(f"{idx}. Long idea: **{b.title[:60]}** — Rationale: recent positive news; consider watch for momentum/volume. Entry: technical pullback; initial target: +8-15%. Risk: verify fundamentals and valuations. Source: {b.url}")
                idx += 1
        if bearish:
            for b in bearish[:3]:
                lines.append(f"{idx}. Short / Avoid: **{b.title[:60]}** — Rationale: negative headlines; watch for regulatory or demand shocks. Consider hedging. Source: {b.url}")
                idx += 1
        if not bullish and not bearish:
            lines.append("1. No obvious directional trade signals found from headlines; consider monitoring earnings and macro data.")
    else:
        lines.append("1. No data to form trade ideas.\n")
    # Risk & caveats
    lines.append("\n## Risk & Caveats\n")
    lines.append("- This report is for informational purposes only. Validate with fundamental data and exchange-level market data before trading.")
    lines.append("- Headlines can be noisy; always check official filings and price/volume confirmations.\n")
    # Watchlist
    lines.append("## Suggested Watchlist / Next Steps\n")
    if news:
        for n in news[:5]:
            lines.append(f"- {n.title} — {n.url}")
    else:
        lines.append("- No items to watch.\n")
    # Sources
    lines.append("\n## Sources\n")
    if news:
        for n in news:
            lines.append(f"- {n.url}")
    else:
        lines.append("- No sources collected.")
    lines.append("\n---\n_Report generated by Trade Opportunities API_")
    return "\n".join(lines)
```

```python name=app/main.py
import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Path
from fastapi.responses import Response
from .models import TokenRequest, TokenResponse
from . import collector, ai_client, auth, rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trade-api")

app = FastAPI(title="Trade Opportunities API", version="0.1.0", docs_url="/docs")

@app.post("/auth/token", response_model=TokenResponse)
async def login_for_token(creds: TokenRequest):
    """
    Obtain a JWT access token using username/password.
    Demo user: dev / devpass
    """
    user = auth.authenticate_user(creds.username, creds.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    data = {"sub": user["username"], "role": user.get("role", "developer")}
    token = auth.create_access_token(data)
    return TokenResponse(access_token=token, expires_in=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))*60)

@app.post("/auth/guest", response_model=TokenResponse)
async def guest_token():
    """
    Create a guest token (limited privileges). No password required.
    """
    from datetime import timedelta
    sub = "guest"
    data = {"sub": sub, "role": "guest"}
    # shorter expiry for guest
    token = auth.create_access_token(data, expires_delta=timedelta(minutes=30))
    return TokenResponse(access_token=token, expires_in=30*60)

@app.get("/analyze/{sector}")
async def analyze_sector(
    sector: str = Path(..., min_length=1, max_length=50, regex=r"^[A-Za-z0-9\s\-\_]+$"),
    token: str = Depends(auth.get_current_token)
):
    """
    Analyze a given sector and return a markdown report.
    Auth: Bearer <token> required
    Rate-limited per session/token.
    """
    # Rate limiting
    try:
        rate_limiter.check_rate_limit(token, auth.SESSIONS)
    except HTTPException as e:
        raise e

    # Data collection
    try:
        news = await collector.collect_market_data(sector)
    except Exception:
        logger.exception("Data collection failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Failed to collect market data")

    # AI analysis
    try:
        markdown = await ai_client.analyze_with_gemini(sector, news)
        if not markdown:
            raise Exception("Empty analysis")
    except Exception:
        logger.exception("AI analysis failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI analysis failed")

    # Return markdown report
    return Response(content=markdown, media_type="text/markdown")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

```text name=requirements.txt
fastapi==0.99.2
uvicorn[standard]==0.22.0
httpx==0.24.1
python-jose==3.3.0
duckduckgo_search==2.7.2
python-dotenv==1.0.0
```

```text name=.env.example
# Example environment variables for Trade Opportunities API
SECRET_KEY=change_this_to_a_strong_random_value
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Optional: configure Gemini (or other LLM) if you have credentials
GEMINI_API_KEY=
GEMINI_API_URL=https://your-gemini-endpoint.example/v1/generate

# Rate limits (per minute)
RATE_LIMIT_AUTH_PER_MINUTE=60
RATE_LIMIT_GUEST_PER_MINUTE=5

# LLM timeout seconds (float)
LLM_TIMEOUT=20.0
```

```text name=.gitignore
__pycache__/
.venv/
.env
.env.local
```

Notes:
- These files form a runnable FastAPI app that meets the single-endpoint requirement. To run locally:
  1. Create and activate a virtualenv (Python 3.10+).
  2. pip install -r requirements.txt
  3. Copy .env.example to .env and set SECRET_KEY.
  4. uvicorn app.main:app --reload --port 8000
- The ai_client will call the configured GEMINI_API_URL when GEMINI_API_KEY & URL are set; otherwise it falls back to a deterministic mock generator (useful for quick testing).
- Authentication is JWT-based and stored in-memory in auth.SESSIONS. Guest tokens are available from /auth/guest.
- Rate limiting is per-token and in-memory.

If you want, I can:
- Add a Dockerfile and GitHub Actions workflow,
- Swap DuckDuckGo scraping for NewsAPI (requires key),
- Wire a production-ready Google Gemini request body if you provide endpoint and auth details.
