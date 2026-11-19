"""Send one tiny request to each model and report which ones your key can use right now.

Usage:  uv run python scripts/check_models.py
        uv run python scripts/check_models.py gemini-2.5-flash gemini-flash-latest
"""

import argparse
import os
import time

DEFAULT_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
        timeout=60,
        max_retries=0,  # one try each, so a failing model doesn't eat quota
    )
    for model in args.models:
        start = time.perf_counter()
        try:
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with: ok"}]
            )
            print(f"OK    {model:<28} {time.perf_counter() - start:5.1f} s")
        except Exception as error:
            message = getattr(error, "message", str(error)).replace("\n", " ")[:150]
            print(f"FAIL  {model:<28} {type(error).__name__}: {message}")
        time.sleep(3)


if __name__ == "__main__":
    main()
