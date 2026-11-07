"""Identify the speaker of one recording; prints 【Name】 (or 【未知】). Used as a Hermes STT command.

identify.py <audio> [--threshold 0.5] [--verbose]
"""

import argparse
import sys

from speaker import embed, load_voiceprints

DEFAULT_THRESHOLD = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    prints = load_voiceprints()
    if not prints:
        print("【未知】")
        return
    try:
        emb = embed(args.audio)
    except Exception as exc:  # too short / undecodable -> unknown, never crash Hermes
        print(f"identify: {exc}", file=sys.stderr)
        print("【未知】")
        return

    scores = sorted(((float(emb @ vec), name) for name, vec in prints.items()), reverse=True)
    if args.verbose:
        for score, name in scores:
            print(f"  {name}: {score:.3f}", file=sys.stderr)
    best_score, best_name = scores[0]
    print(f"【{best_name}】" if best_score >= args.threshold else "【未知】")


if __name__ == "__main__":
    main()
