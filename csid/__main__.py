"""One command per thing a reader might want to check.

    python -m csid doctor         the data, the kinematics, and whether they agree
    python -m csid identifiability  can this recording support the parameters at all
    python -m csid library        smooth vs non-smooth, fitted and crossed over
    python -m csid smoothness     how much the answer depends on the smoothing window
    python -m csid all            everything above, written to bench/results.json

Nothing here prints a number without the sample size and the regime it came from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def _finite(obj):
    """Replace non-finite floats with strings before writing.

    `json.dump` writes inf and nan as the bare tokens Infinity and NaN, which the JSON
    standard does not define and other parsers need not accept; Python's own reader is
    lenient, which is exactly why this is easy to miss. `allow_nan=False` turns a
    recurrence into an error instead of a file nobody else can read.
    """
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj in (float("inf"), float("-inf")):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _dump(obj, name):
    os.makedirs(os.path.join(BASE, "bench"), exist_ok=True)
    p = os.path.join(BASE, "bench", name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_finite(obj), f, indent=1, allow_nan=False)
    return p


def cmd_doctor(args):
    from . import data as D, kinematics as K

    print("dataset:", D.DATA)
    m = D.check_fields()
    print("fps:", m["fps"], " episodes:", m["episodes"],
          " state:", m["state"]["shape"], m["state"]["names"])
    print("wrench:", m["wrench"]["shape"], m["wrench"]["names"])
    print()
    for r in D.summary():
        print("ep%d  %4d frames  %5.1f s  dt=%.5f s  |F| median %6.2f p90 %6.2f max %7.2f N"
              % (r["episode"], r["frames"], r["seconds"], r["dt_median_s"],
                 r["force_N"]["median"], r["force_N"]["p90"], r["force_N"]["max"]))
    print()
    for i, ts, q, w in D.episodes():
        s = K.workspace_stats(q.tolist())
        print("ep%d  tcp travel %s m  radius %.3f..%.3f m  kinematic ok: %s  "
              "working envelope: %s"
              % (i, s["tcp_travel_m"], s["tcp_radius_min_m"], s["tcp_radius_max_m"],
                 s["within_kinematic_reach"], s["within_working_envelope"]))
    return 0


def cmd_identifiability(args):
    from . import study

    ed = study.load_arrays()
    r = study.identifiability(ed)
    print("Can tool mass and sensor bias be told apart in this recording?")
    print()
    for e in r["per_episode"]:
        print("ep%d  cond(A)=%9.0f  gravity spread %5.3f%%  mass(restricted) %8.2f +- %7.2f kg"
              "  t=%6.2f  mass(unrestricted) %9.1f kg"
              % (e["episode"], e["cond"], 100 * e["gravity_spread_frac"],
                 e["scalar_mass_kg"], e["scalar_mass_stderr_kg"], e["scalar_mass_t"],
                 e["unrestricted_mass_kg"]))
    p = r["pooled"]
    print()
    print("pooled over %d samples: cond=%.1f  sigma_min=%.2e  identifiable=%s"
          % (p["n_samples"], p["cond"], p["sigma_min"], p["identifiable"]))
    ps = r["pooled_scalar_mass"]
    print()
    print("the mass the whole recording supports, restricted to one shared scalar:")
    print("  m = %.2f +- %.2f kg  95%% CI %s  t=%.2f  identified=%s"
          % (ps["mass_kg"], ps["stderr_kg"], ps["ci95_kg"], ps["t_stat"], ps["identified"]))
    print("  %s" % ps["reason"])
    inj = r["inert_joints"]
    print("  joints that cannot change the gravity direction at ANY configuration: %s"
          % ([j + 1 for j in inj["structural"]] or "none"))
    print("  joints that happen to be inert at some configurations: %s"
          % ([j + 1 for j in inj["situational"]] or "none"))
    q = r["excitation_required"]
    print()
    print("to reach cond<=%.0e the gravity direction's variation would need to grow %.1fx"
          % (q["target_cond"], q["excitation_factor_needed"]))
    print("  joints where more motion would help: %s" % q["excitable_joints"])
    print("  recorded joint ranges (rad):", q["current_joint_range_rad"])
    print("  joints barely moving: %d" % q["joints_barely_moving"])
    print("  joints that cannot excite gravity at all: %s"
          % ([j + 1 for j in q["joints_that_cannot_excite_gravity_at_all"]] or "none"))
    print()
    print("reading:", q["reading"])
    _dump(r, "identifiability.json")
    return 0


def cmd_library(args):
    from . import study
    import json as _j

    ed = study.load_arrays()
    out = {"smooth": study.cross_table(ed, lam=args.lam, library="smooth"),
           "nonsmooth": study.cross_table(ed, lam=args.lam, library="nonsmooth")}
    for name in ("smooth", "nonsmooth"):
        t = out[name]
        print("%s library: %d terms, lambda=%s, free<=%.1f N (%d rows), contact>=%.1f N (%d rows)"
              % (name, t["n_terms"], t["lambda"], t["regimes"]["free_max_N"], t["regimes"]["n_free"],
                 t["regimes"]["contact_min_N"], t["regimes"]["n_contact"]))
        for k, v in t["fits"].items():
            print("   %-32s nonzero %4d   R2 %+.4f" % (k, v["nonzero"], v["r2_mean"]))
        print("   group mass (all data):", _j.dumps(t["group_mass_all_data"]))
        print()
    _dump(out, "library.json")
    return 0


def cmd_contact(args):
    from . import study, library as L

    ed = study.load_arrays()
    dummy = ([[0.0] * 6] * 2, [[0.0] * 6] * 2, [[0.0] * 6] * 2)
    terms = {"smooth": len(L.smooth_library(*dummy)[1]),
             "nonsmooth": len(L.nonsmooth_library(*dummy)[1])}
    print("Within each episode: fit on the frames below that episode's own %.0fth force"
          % args.q)
    print("percentile, predict the frames at or above it. Same motion, same tool, so")
    print("the episodes' different force baselines cannot explain the result.")
    print()
    for lib in ("smooth", "nonsmooth"):
        rows = study.within_episode_contact(ed, lam=args.lam, library=lib, contact_q=args.q)
        print("%s library (%d terms)" % (lib, terms[lib]))
        print("  ep   thr(N)   n_free  n_contact    R2_free    R2_contact_heldout")
        for r in rows:
            print("  %2d  %7.2f  %6d  %9d   %+8.4f   %+18.4f"
                  % (r["episode"], r["contact_threshold_N"], r["n_train_free"],
                     r["n_held_out_contact"], r["r2_train_free"], r["r2_held_out_contact"]))
        print()
    for lib in ("smooth", "nonsmooth"):
        rows = study.leave_one_episode_out(ed, lam=args.lam, library=lib)
        print("leave one whole operation out (%s):" % lib)
        for r in rows:
            print("  hold out ep%d  train %5d rows  test %5d rows   R2 %+.4f   per-axis %s"
                  % (r["held_out_episode"], r["n_train"], r["n_test"], r["r2_mean"],
                     r["r2_per_axis"]))
        print()
    return 0


def cmd_smoothness(args):
    from . import study

    rows = study.smoothing_sensitivity()
    print("sensitivity of the free-motion fit to the Savitzky-Golay window")
    print("  window   R2_free   R2_contact")
    for r in rows:
        print("  %6d   %+.4f   %+10.4f" % (r["window"], r["r2_free"], r["r2_contact"]))
    _dump(rows, "smoothing.json")
    return 0


def cmd_all(args):
    from . import study

    r = study.run(lam=args.lam)
    p = _dump(r, "results.json")
    print("wrote", p)
    print()
    print("within-episode: fit on free frames, predict the held-out contact frames")
    print("  %-11s %5s %11s %18s %14s" % ("library", "ep", "R2_free", "R2_contact_out", "R2_axial_fz"))
    for lib, rows in r["library_comparison"].items():
        for x in rows:
            print("  %-11s %5d %+11.4f %+18.4f %+14.4f"
                  % (lib, x["episode"], x["r2_train_free"], x["r2_held_out_contact"],
                     x["r2_axial_force"]))
    print()
    print("pooled cond(A) = %.1f  (mass and bias not separable; identifiable=%s)"
          % (r["identifiability"]["pooled"]["cond"],
             r["identifiability"]["pooled"]["identifiable"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="csid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("doctor", cmd_doctor), ("identifiability", cmd_identifiability),
                     ("library", cmd_library), ("contact", cmd_contact),
                     ("smoothness", cmd_smoothness), ("all", cmd_all)):
        sp = sub.add_parser(name)
        if name in ("library", "contact", "all"):
            sp.add_argument("--lam", type=float, default=0.05)
        if name == "contact":
            sp.add_argument("--q", type=float, default=90.0)
        sp.set_defaults(func=fn)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
