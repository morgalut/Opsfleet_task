# main.py
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.helper_agent.config import AppConfig, load_config
from src.helper_agent.graph import create_agent_app
from src.helper_agent.models import AgentState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph Helper Agent - LangGraph/LangChain Q&A assistant (with ORC/ReAct/hybrid + debug)."
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        help="Agent operating mode. Overrides AGENT_MODE env var.",
    )
    parser.add_argument(
        "--strategy",
        choices=["basic", "orc", "react", "hybrid"],
        help=(
            "Reasoning strategy: "
            "basic = original RAG, "
            "orc = ORC-style planning, "
            "react = ReAct loop, "
            "hybrid = ORC entry (you can extend).\n"
            "Defaults to $AGENT_REASONING or 'basic'."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Show internal reasoning (ORC plan, subquestions, ReAct scratchpad, "
            "retrieved_docs, iterations, etc.) and save it to reasoning_trace.json."
        ),
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The question to ask the helper agent.",
    )
    return parser.parse_args()


def _print_section(title: str, content: Any) -> None:
    """Pretty-print a debug section."""
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    if isinstance(content, str):
        print(content)
    else:
        # Fallback pretty JSON if it's complex
        try:
            print(json.dumps(content, indent=2, ensure_ascii=False))
        except Exception:
            print(repr(content))
    print("\n")


def dump_debug_state(final_state: AgentState) -> None:
    """
    Print and save the full internal reasoning state:
      - ORC: orc_plan, orc_subquestions
      - ReAct: scratchpad, react_query, iterations
      - retrieval: retrieved_docs
      - and any other keys present
    """
    print("\n\n================ DEBUG: INTERNAL REASONING TRACE ================\n")

    # Print key sections in a sane order
    ordered_keys = [
        "question",
        "mode",
        "reasoning_style",
        "orc_plan",
        "orc_subquestions",
        "retrieved_docs",
        "scratchpad",
        "react_query",
        "iterations",
        "error",
    ]

    printed = set()

    for key in ordered_keys:
        if key in final_state and key != "final_answer":
            _print_section(key.upper(), final_state[key])
            printed.add(key)

    # Print any other keys that weren't in the ordered list
    for key, value in final_state.items():
        if key in printed or key == "final_answer":
            continue
        _print_section(key.upper(), value)

    # Save everything to JSON file
    try:
        with open("reasoning_trace.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2, ensure_ascii=False, default=str)
        print("Debug reasoning saved to reasoning_trace.json\n")
    except Exception as e:
        print(f"[WARN] Could not save reasoning trace: {e}\n")


def main() -> None:
    args = parse_args()

    if not args.question:
        print("Please provide a question, e.g.:")
        print(
            "  python main.py --mode offline --strategy orc "
            "--debug \"How do I add persistence to a LangGraph agent?\""
        )
        sys.exit(1)

    question = " ".join(args.question)

    # Load configuration (mode + reasoning style)
    config: AppConfig = load_config(
        cli_mode=args.mode,
        cli_reasoning=args.strategy,
    )

    app = create_agent_app(config)

    # Initial LangGraph state
    initial: AgentState = {
        "question": question,
        "mode": config.mode,
        "reasoning_style": config.reasoning_style,
    }

    # Run LangGraph agent
    final: AgentState = app.invoke(initial)  # type: ignore

    # Debug: show internal reasoning + save JSON
    if args.debug:
        dump_debug_state(final)

    # Always print final answer cleanly at the end
    print("\n=== Answer ===\n")
    print(final.get("final_answer", "(no answer)"))


if __name__ == "__main__":
    main()
