import time
import random
import traceback
from typing import Callable, Tuple, Type

from google.api_core.exceptions import ResourceExhausted

def is_retryable_error(e: Exception) -> bool:
    msg = str(e).lower()

    retryable_text = (
        "resource exhausted",
        "429",
        "too many requests",
        "exceeded the provisioned throughput",
        "quota",
        "capacity",
        "temporarily unavailable",
    )

    if any(t in msg for t in retryable_text):
        return True

    # For Google exceptions:
    from google.api_core.exceptions import (
        ResourceExhausted,
        ServiceUnavailable,
        InternalServerError,
        DeadlineExceeded,
    )

    return isinstance(e, (
        ResourceExhausted,
        ServiceUnavailable,   # 503
        InternalServerError,  # 500
        DeadlineExceeded,     # 504
    ))


def with_rate_limit_retry(
    fn,
    max_attempts=6,
    base_delay=1.0,
    max_delay=60.0,
    node_name=None,
):
    name = node_name or getattr(fn, "__name__", "node")

    def wrapped(state):
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(state)

            except Exception as e:
                # check if retryable
                if not is_retryable_error(e):
                    print(f"[Retry:{name}] NON-RETRYABLE → raising: {e}")
                    raise

                print(f"[Retry:{name}] attempt {attempt}/{max_attempts} → retryable error: {e}")

                # Check Retry-After header (Vertex usually includes it)
                retry_after = getattr(e, "retry_after", None)
                if retry_after:
                    delay = retry_after
                else:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay += random.uniform(0, 0.5)

                print(f"[Retry:{name}] sleeping {delay:.2f}s…")
                time.sleep(delay)

                if attempt == max_attempts:
                    new_state = dict(state)
                    new_state["error"] = f"{name} failed after {max_attempts} attempts: {e}"
                    return new_state

        return state

    return wrapped

