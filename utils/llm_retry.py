import re
import time


def _status_code(exc: Exception) -> int | None:
    """Pull an HTTP status code out of Groq / httpx / LangChain exceptions."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        return response.status_code
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    """Groq sends a retry-after header, and its message often says 'try again in 12.3s'."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


def is_rate_limit(exc: Exception) -> bool:
    return _status_code(exc) == 429 or "rate limit" in str(exc).lower()


def invoke_with_retry(chain, input_data, max_retries: int = 4, base_wait: float = 15):
    """Invoke a LangChain runnable, retrying when Groq rate-limits us (HTTP 429).

    Groq's free tier limits tokens per minute, so we wait for the time Groq asks
    for (retry-after), falling back to a growing backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it's a 429
            if not is_rate_limit(exc) or attempt == max_retries:
                raise
            wait = _retry_after_seconds(exc) or base_wait * attempt
            wait = min(wait + 1, 90)
            print(f"Rate limited by Groq (attempt {attempt}/{max_retries}); waiting {wait:.0f}s...")
            time.sleep(wait)
