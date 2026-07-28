"""Job verifier: cross-check listings via alternate source using Firecrawl SDK."""

from firecrawl import FirecrawlApp

from src.llm.context import ContextManager

VERIFY_PROMPT = """Compare two scraped job listings. Are they the same job?
Original: {original}
Alternate: {alternate}
Return ONLY JSON: {{"same_job": true/false, "confidence": 0-100}}"""


def verify_job(
    app: FirecrawlApp,
    role: str,
    company: str,
    original_url: str,
    ctx: ContextManager,
) -> bool:
    try:
        alt_results = app.search(f"{role} {company} job")
        data = alt_results.get("data", alt_results)
        if not isinstance(data, list) or not data:
            return True

        alt_url = None
        for r in data:
            u = r.get("url", r.get("metadata", {}).get("url", ""))
            if u and u != original_url and u.startswith("http"):
                alt_url = u
                break

        if not alt_url:
            return True

        alt_result = app.scrape_url(alt_url, formats=["markdown"])
        alt_content = alt_result.get("markdown", alt_result.get("data", {}).get("markdown", ""))
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
        return True
