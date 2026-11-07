"""Enroll a speaker: enroll.py --name Josh a.wav b.wav ..."""

import argparse

import numpy as np

from speaker import embed, load_voiceprints, save_voiceprints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    embs = [embed(f) for f in args.files]
    mean = np.mean(embs, axis=0)
    mean /= np.linalg.norm(mean)
    for f, e in zip(args.files, embs):
        print(f"{f}: similarity to {args.name} centroid = {float(e @ mean):.3f}")

    prints = load_voiceprints()
    prints[args.name] = mean
    save_voiceprints(prints)
    print(f"Enrolled {args.name} from {len(embs)} clips. Speakers: {', '.join(prints)}")


if __name__ == "__main__":
    main()
