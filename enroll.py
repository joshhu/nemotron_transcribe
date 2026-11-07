"""Enroll a speaker.

enroll.py --name Josh a.wav b.wav ...        replace Josh's voiceprints with one centroid of these clips
enroll.py --name Sandy --add laptop1.webm    add another centroid (e.g. clips from a different mic)
"""

import argparse

import numpy as np

from speaker import embed, load_voiceprints, save_voiceprints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--add", action="store_true",
                    help="add a centroid instead of replacing (one per mic / recording condition)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    embs = [embed(f) for f in args.files]
    mean = np.mean(embs, axis=0)
    mean /= np.linalg.norm(mean)
    for f, e in zip(args.files, embs):
        print(f"{f}: similarity to new {args.name} centroid = {float(e @ mean):.3f}")

    prints = load_voiceprints()
    if args.add and args.name in prints:
        prints[args.name] = np.vstack([prints[args.name], mean])
    else:
        prints[args.name] = mean[None]
    save_voiceprints(prints)
    print(f"{args.name}: {len(prints[args.name])} centroid(s) from {len(embs)} new clip(s). "
          f"Speakers: {', '.join(prints)}")


if __name__ == "__main__":
    main()
