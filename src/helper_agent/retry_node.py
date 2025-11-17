# src/helper_agent/retry_node.py
from __future__ import annotations

import time
import random
import traceback
from dataclasses import dataclass
from typing import Callable, Tuple, Type

from .models import AgentState

ExceptionTypes = Tuple[Type[BaseException], ...]


@dataclass
class RetryNode:


    base_node: Callable[[AgentState], AgentState]
    name: str
    max_attempts: int = 3
    base_delay: float = 1.0
    exception_types: ExceptionTypes = (Exception,)

    def __call__(self, state: AgentState) -> AgentState:
        last_exc: BaseException | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.base_node(state)
            except self.exception_types as e:  # type: ignore[misc]
                last_exc = e
                print(
                    f"[RetryNode:{self.name}] attempt {attempt}/{self.max_attempts} failed: {e}"
                )
                traceback.print_exc()

                if attempt == self.max_attempts:
                    new_state = dict(state)
                    new_state["error"] = (
                        f"{self.name} failed after {self.max_attempts} attempts: {e}"
                    )
                    return new_state  # type: ignore[return-value]

                delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(f"[RetryNode:{self.name}] sleeping {delay:.2f}s before retry...")
                time.sleep(delay)

        return state

    def as_node(self) -> Callable[[AgentState], AgentState]:
        """
        Convenience method to pass directly to builder.add_node.
        """
        return self.__call__
