# src/helper_agent/retry_utils.py
from __future__ import annotations

import time
import random
import traceback
from typing import Callable, Tuple, Type

from .models import AgentState

ExceptionTypes = Tuple[Type[BaseException], ...]


def with_retry(
    fn: Callable[[AgentState], AgentState],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exception_types: ExceptionTypes = (Exception,),
    node_name: str | None = None,
) -> Callable[[AgentState], AgentState]:
    """
    Wrap a LangGraph node function with retry logic.

    - fn: node(state) -> state
    - max_attempts: max times to try
    - base_delay: initial delay for exponential backoff
    - exception_types: which exceptions to retry
    - node_name: for logging (defaults to fn.__name__)
    """

    name = node_name or getattr(fn, "__name__", "node")

    def wrapped(state: AgentState) -> AgentState:
        last_exc: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return fn(state)
            except exception_types as e:  # type: ignore[misc]
                last_exc = e
                print(
                    f"[Retry:{name}] attempt {attempt}/{max_attempts} failed: {e}"
                )
                traceback.print_exc()

                if attempt == max_attempts:
                    # attach error info to state and return
                    new_state = dict(state)
                    new_state["error"] = (
                        f"{name} failed after {max_attempts} attempts: {e}"
                    )
                    return new_state  # type: ignore[return-value]

                # exponential backoff + jitter
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(f"[Retry:{name}] sleeping {delay:.2f}s before retry...")
                time.sleep(delay)

        # Should not reach here, but keep mypy happy:
        return state

    return wrapped
