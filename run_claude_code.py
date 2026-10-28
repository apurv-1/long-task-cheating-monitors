"""Run Claude Code on METR tasks, then scrub the subscription token from every log.

    export CLAUDE_CODE_OAUTH_TOKEN=...   # from `claude setup-token`, or put it in .env
    uv run python run_claude_code.py \
        --image-tag registry-1.docker.io/metrevals/public-tasks:cowthello-2.0.2 \
        --sample-id main --hours 2

Use this script instead of `inspect eval`, so no log is left unscrubbed.
"""

import argparse
import os

from dotenv import load_dotenv
from inspect_ai import eval
from mtb.bridge import bridge

from claude_code_agent import TOKEN_ENV, claude_code, scrub_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-tag", required=True, help="Full METR task image name.")
    parser.add_argument("--sample-id", required=True, help="Task name within the family.")
    parser.add_argument("--hours", type=float, default=2.0, help="Wall-clock limit per run.")
    parser.add_argument("--model", default=None, help="Claude Code --model value, e.g. opus.")
    parser.add_argument("--version", default="stable", help="Claude Code version to install.")
    parser.add_argument("--epochs", type=int, default=1, help="Repeated runs of the task.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--install-command",
        default=None,
        help="Replace the official installer. Only for testing with a fake binary.",
    )
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"Set {TOKEN_ENV} from `claude setup-token` first.")

    logs = []
    try:
        logs = eval(
            bridge(image_tag=args.image_tag),
            solver=claude_code(
                timeout_hours=args.hours,
                model=args.model,
                version=args.version,
                install_command=args.install_command,
            ),
            model="none",
            sample_id=args.sample_id,
            epochs=args.epochs,
            max_samples=1,
            log_dir=args.log_dir,
        )
    finally:
        # Scrub even when the run fails, since partial logs can hold the token too.
        for path in _logs_to_scrub(logs, args.log_dir):
            scrub_log(path, token)
            print(f"Scrubbed {path}")


def _logs_to_scrub(logs, log_dir: str) -> list[str]:
    paths = [log.location for log in logs if log.location]
    if paths:
        return paths
    # eval() raised before returning, so scrub every log in the directory.
    return [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.endswith((".eval", ".json"))
    ]


if __name__ == "__main__":
    main()
