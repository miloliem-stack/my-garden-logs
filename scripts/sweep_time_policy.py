import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy_replay import legacy_policy_replay_message


def main() -> int:
    raise SystemExit(legacy_policy_replay_message())


if __name__ == "__main__":
    main()
