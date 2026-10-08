"""Does the repository still produce the numbers it publishes?

Separate from test_csid.py on purpose. That file tests conventions and mechanisms on
synthetic data, so it keeps working when the study is re-run on other data. This file is
the opposite: it compares the current run against bench/results.json and fails if the
published numbers have drifted. Two different jobs, two different files -- a single file
doing both would have to choose which failure it reports.

Tolerances are absolute and generous enough for a different BLAS, because the point is to
catch a changed *conclusion* (an R2 that flips sign, a condition number that moves by an
order of magnitude), not to pin the last digit of a floating-point sum.

Run:  python tests/test_reproducible.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from csid import study  # noqa: E402

TOL_R2 = 0.05
TOL_COND_REL = 0.2


def _load():
    p = os.path.join(BASE, "bench", "results.json")
    if not os.path.exists(p):
        print("bench/results.json is absent; run `python -m csid all` first")
        raise SystemExit(2)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    published = _load()
    now = study.run()
    bad = []

    for lib, rows in published["library_comparison"].items():
        for r in rows:
            k = r["episode"]
            cur = [x for x in now["library_comparison"][lib] if x["episode"] == k]
            if not cur:
                bad.append(f"{lib} ep{k}: gone from the run")
                continue
            cur = cur[0]
            for field in ("r2_train_free", "r2_held_out_contact", "r2_axial_force"):
                if abs(cur[field] - r[field]) > TOL_R2:
                    bad.append(f"{lib} ep{k} {field}: published {r[field]:+.4f} "
                               f"now {cur[field]:+.4f}")

    a = published["identifiability"]["pooled_scalar_mass"]
    b = now["identifiability"]["pooled_scalar_mass"]
    for field in ("mass_kg", "stderr_kg", "t_stat"):
        if abs(b[field] - a[field]) > max(0.05 * abs(a[field]), 0.05):
            bad.append(f"pooled {field}: published {a[field]} now {b[field]}")
    if a["identified"] != b["identified"]:
        bad.append("pooled identified changed: the study's headline claim moved")

    pc = published["identifiability"]["pooled"]["cond"]
    nc = now["identifiability"]["pooled"]["cond"]
    if abs(nc - pc) > TOL_COND_REL * pc:
        bad.append(f"pooled cond: published {pc:.0f} now {nc:.0f}")

    pi = published["identifiability"]["inert_joints"]["structural"]
    ni = now["identifiability"]["inert_joints"]["structural"]
    if pi != ni:
        bad.append(f"structural inert joints changed: {pi} -> {ni}")

    # the sign of the finding, asserted as a sign and not as a magnitude
    if b["identified"] or min(b["ci95_kg"]) > 0:
        bad.append("the mass is now identifiable from this recording -- either the data "
                   "or the kinematics changed, and the README's first finding is stale")
    hist = now["library_comparison"]["history"]
    smooth = now["library_comparison"]["smooth"]
    h_axial = max(x["r2_axial_force"] for x in hist)
    s_axial = max(x["r2_axial_force"] for x in smooth)
    if h_axial <= s_axial:
        bad.append("the force-history control no longer beats the motion-only library on "
                   "the axial force, which is the second finding's whole evidence")

    if bad:
        print("the published numbers no longer match the run:")
        for x in bad:
            print("  -", x)
        return 1
    print("reproduced: %d library rows, pooled mass %.2f +- %.2f kg, cond %.0f"
          % (sum(len(v) for v in now["library_comparison"].values()),
             b["mass_kg"], b["stderr_kg"], nc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
