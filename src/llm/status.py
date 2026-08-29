"""Inspect LLM configuration or make explicit smoke-test calls."""

from __future__ import annotations

import argparse
import asyncio
import os

from .router import Tier, call_llm, embed


FIELDS = ("API_KEY", "BASE_URL", "MODEL")


def status() -> int:
    for tier in Tier:
        missing = [f"{tier.value}_{field}" for field in FIELDS if not os.getenv(f"{tier.value}_{field}")]
        result = "configured" if not missing else f"missing: {', '.join(missing)}"
        print(f"{tier.value.lower():10} {result}")
    return 0


async def smoke() -> int:
    failures = 0
    for tier in Tier:
        try:
            if tier is Tier.EMBED:
                await embed(tier, ["ping"])
            else:
                await call_llm(tier, [{"role": "user", "content": "ping"}], max_tokens=1)
            print(f"{tier.value.lower():10} ok")
        except Exception as error:  # A CLI should report every tier, not stop at the first failure.
            failures += 1
            print(f"{tier.value.lower():10} failed: {error}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "smoke"), nargs="?", default="status")
    command = parser.parse_args().command
    return status() if command == "status" else asyncio.run(smoke())


if __name__ == "__main__":
    raise SystemExit(main())
