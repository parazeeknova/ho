"""Context manager: token tracking, auto-flush, retry wrapper for LLM calls."""

import json
import time
import urllib.request

LLM_URL = "http://127.0.0.1:8899"
MODEL = "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:Q5_K_M"
MAX_RETRIES = 3
RETRY_DELAY = 4
TOKEN_ESTIMATE_PER_CHAR = 0.4
FLUSH_THRESHOLD = 6000  # estimated tokens — flush every ~6K tokens of cumulative output


class ContextManager:
    def __init__(self) -> None:
        self.cumulative_output_tokens = 0

    def chat(self, prompt: str) -> str:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = json.dumps(
                    {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    }
                ).encode()
                req = urllib.request.Request(
                    f"{LLM_URL}/v1/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                    output = result["choices"][0]["message"]["content"]
                    tokens = int(len(output) * TOKEN_ESTIMATE_PER_CHAR)
                    self.cumulative_output_tokens += tokens
                    return output
            except Exception:
                if attempt < MAX_RETRIES:
                    print(f"  [LLM retry {attempt}/{MAX_RETRIES}]")
                    time.sleep(RETRY_DELAY)
        raise RuntimeError("LLM failed after all retries")

    def maybe_flush(self) -> None:
        if self.cumulative_output_tokens >= FLUSH_THRESHOLD:
            self.flush()

    def flush(self) -> None:
        try:
            slots = json.loads(urllib.request.urlopen(f"{LLM_URL}/slots", timeout=5).read())
            for slot in slots:
                sid = slot.get("id")
                if sid is not None and slot.get("state") != 0:
                    urllib.request.urlopen(f"{LLM_URL}/slots/{sid}?action=erase", timeout=5)
            self.cumulative_output_tokens = 0
            print("  [ctx flushed]")
        except Exception:
            pass

    def clean_json(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        return raw.strip()

    def json_chat(self, prompt: str, content: str = "", limit: int = 28000) -> dict | list:
        full = prompt
        if content:
            full = prompt + "\n\n" + content[:limit]
        raw = self.chat(full)
        try:
            return json.loads(self.clean_json(raw))
        except json.JSONDecodeError:
            return {} if "{" in prompt else []
