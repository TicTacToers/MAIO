from __future__ import annotations

import json
import math
import sys
from pathlib import Path

IGNORED = {"train_seconds"}
ABS_TOL = 1e-3


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=ABS_TOL)
    return a == b


def main(ci_path: str, image_path: str) -> int:
    ci = json.loads(Path(ci_path).read_text())
    image = json.loads(Path(image_path).read_text())
    keys = sorted((ci.keys() | image.keys()) - IGNORED)
    diffs = [
        f"{k}: CI={ci.get(k)!r} image={image.get(k)!r}"
        for k in keys
        if not _same(ci.get(k), image.get(k))
    ]
    if diffs:
        print("Image model differs from the CI training run:")
        print("\n".join(f"  {d}" for d in diffs))
        return 1
    print(f"Image model matches the CI training run ({len(keys)} fields compared).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))

