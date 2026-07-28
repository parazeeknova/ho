"""Job verifier: cross-check listings via alternate source."""

import json
import urllib.request

FC_URL = "http://127.0.0.1:3002"

VERIFY_PROMPT = """Compare two scraped job listings. Are they the same job?
Original: {original}
Alternate: {alternate}
Return ONLY JSON: {{"same_job": true/false, "confidence": 0-100}}"""


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def scrape(url: str) -> str:
    result = _post(f"{FC_URL}/v1/scrape", {"url": url, "formats": ["markdown"]})
    return result["data"]["markdown"]


def search(query: str) -> list[dict]:
    return _post(f"{FC_URL}/v1/search", {"query": query}).get("data", [])


def verify_job(role: str, company: str, original_url: str, ctx) -> bool:
    """Search for the same job on another platform, compare."""
    try:
        alt_results = search(f"{role} {company} job")
        if not alt_results:
            return True  # can't verify, assume valid

        alt_url = None
        for r in alt_results:
            u = r.get("url", "")
            if u and u != original_url and u.startswith("http"):
                alt_url = u
                break

        if not alt_url:
            return True

        alt_content = scrape(alt_url)
        if len(alt_content) < 100:
            return True

        prompt = VERIFY_PROMPT.replace("{original}", f"{role} @ {company} [{original_url}]")
        prompt = prompt.replace("{alternate}", alt_content[:3000])

        result = ctx.json_chat(prompt)
        if isinstance(result, dict):
            confidence = result.get("confidence", 50)
            return confidence >= 30

        return True
    except Exception:
        return True  # verification failed, keep the job (don't false-negative)
