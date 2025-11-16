# main.py
import argparse
import sys

from src.helper_agent.config import AppConfig, load_config
from src.helper_agent.graph import create_agent_app, run_agent_once



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph Helper Agent - LangGraph/LangChain Q&A assistant."
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        help="Agent operating mode. Overrides AGENT_MODE env var.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The question to ask the helper agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.question:
        print("Please provide a question, e.g.:")
        print("  python main.py --mode offline 'How do I add persistence to a LangGraph agent?'")
        sys.exit(1)

    question = " ".join(args.question)

    config: AppConfig = load_config(cli_mode=args.mode)
    app = create_agent_app(config)

    answer = run_agent_once(app=app, config=config, question=question)

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
