import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.hmm_visuals import build_regime_overlay_html


def main():
    parser = argparse.ArgumentParser(description="Plot BTC OHLCV with HMM regime overlays.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--title", default="BTC HMM Regime Overlay")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--state-column", default="regime_policy_state")
    parser.add_argument("--confidence-column", default="hmm_map_confidence")
    args = parser.parse_args()
    path = build_regime_overlay_html(
        args.input,
        args.output,
        png_output=args.png_output,
        start=args.start,
        end=args.end,
        title=args.title,
        max_rows=args.max_rows,
        state_column=args.state_column,
        confidence_column=args.confidence_column,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

