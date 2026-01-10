import argparse
from src.agent import run_query


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User analysis request"
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Dataset path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max-new-tokens-plan",
        type=int,
        default=256
    )

    parser.add_argument(
        "--max-new-tokens-tool",
        type=int,
        default=256
    )

    parser.add_argument(
        "--max-new-tokens-final",
        type=int,
        default=512
    )

    args = parser.parse_args()

    result = run_query(
        user_query=args.query,
        dataset_path=args.path,
        verbose=args.verbose,
        max_steps=args.max_steps,
        max_new_tokens_plan=args.max_new_tokens_plan,
        max_new_tokens_tool=args.max_new_tokens_tool,
        max_new_tokens_final=args.max_new_tokens_final,
    )

    print(f"\n{result}")


if __name__ == "__main__":
    main()

