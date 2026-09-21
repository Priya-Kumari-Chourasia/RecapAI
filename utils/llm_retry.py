import time

import httpx


def invoke_with_retry(chain, input_data, max_retries: int = 4, base_wait: int = 65):
    """Invoke a LangChain chain, retrying on 429 rate-limit errors.

    Mistral's free-tier limits are commonly per-minute, so the wait is long
    enough (65s, growing per attempt) for that window to reset rather than
    the few-second backoff langchain's own retry logic uses by default.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except httpx.HTTPStatusError as e:
            last_error = e
            is_rate_limit = e.response is not None and e.response.status_code == 429
            if is_rate_limit and attempt < max_retries:
                wait = base_wait * attempt
                print(
                    f"Rate limited by Mistral (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait}s before retrying..."
                )
                time.sleep(wait)
                continue
            raise
    raise last_error
