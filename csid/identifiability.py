"""Can the excitation in the data even support the model being fitted?

This module exists because of a specific, repeatable failure: fitting tool mass and
force-sensor bias to these four episodes returns numbers in the hundreds of
kilograms. The obvious readings -- a bug, a units error, bad data -- are all wrong.
The correct reading is that the model has more free parameters than the motion can
tell apart, and that is a property of the *experiment*, not of the estimator.

The formal content is one line long. The sensor reads

    f_measured = m * R(q)^T g + b + (everything else)

so `m` and `b` are identified separately only if the column `R(q)^T g` varies
independently of the constant column. The measure of "independently" is the
condition number of [R^T g | 1]; when the tool barely reorients, the two columns are
nearly parallel and the solve splits one effect between two names arbitrarily.

Three functions answer the question at three strengths, and all three are published
because they disagree in an informative way:

  condition()           is the design badly conditioned? (a yes/no about the matrix)
  fit_mass_scalar()     what mass does the recording support, with an error bar?
                        (a number and an interval, and on this data the interval
                        contains zero -- which is the honest form of "not
                        identifiable" and the one that survives a referee)
  excitation_required() how much more motion would fix it? (advice, in units of
                        joint range, and in the joint numbers where motion helps)

The practical consequence is the interesting part: "collect more data" becomes a
number a lab can budget for, and the answer to "how many operations is enough" is
that no number of operations is enough unless the *motion* changes.
"""

from __future__ import annotations

import math
import random

G_WORLD = (0.0, 0.0, -9.81)

#: Beyond this condition number the gravity direction is too nearly constant for the
#: mass and the bias to be assigned to different columns. Declared, not tuned: it is
#: the same number `excitation_required` is asked to reach, so the two functions agree
#: by construction rather than by coincidence.
IDENTIFIABLE_COND = 1e3


def gravity_columns(qs):
    """R(q)^T g for each configuration -- the direction gravity points, as seen by
    the tool. numpy if present."""
    from . import kinematics as K

    rows = []
    for q in qs:
        R = K.rotation(q)
        g = G_WORLD
        rows.append([sum(R[k][i] * g[k] for k in range(3)) for i in range(3)])
    try:
        import numpy as np

        return np.asarray(rows, dtype=float)
    except ImportError:
        return rows


def design(qs):
    """[R^T g | 1] -- the regressors of the mass-plus-bias model."""
    Rg = gravity_columns(qs)
    n = len(qs)
    try:
        import numpy as np

        return np.hstack([Rg, np.ones((n, 1))])
    except ImportError:
        return [list(row) + [1.0] for row in Rg]


def condition(qs) -> dict:
    """How badly the mass/bias split is conditioned, and how much the gravity direction
    moves across the recording."""
    import numpy as np

    A = design(qs)
    s = np.linalg.svd(A, compute_uv=False)
    smax, smin = float(s[0]), float(s[-1])
    Rg = gravity_columns(qs)
    spread = [float(v) for v in (Rg.max(0) - Rg.min(0))]
    # The criterion is the one `excitation_required` aims at, so "is it identifiable"
    # and "how much more motion would make it so" cannot disagree. The threshold is on
    # the condition number rather than on sigma_min because sigma_min is not scale
    # invariant: the gravity column carries ~9.8 and the bias column carries 1, so
    # sigma_min alone moves when the units do.
    # The degeneracy test is physical rather than numerical: `gravity_moves` asks whether
    # the gravity direction varies across the recording *at all*. Testing the condition
    # number instead is a trap, because a design that is under-excited but not constant
    # -- one joint swinging through a third of a radian -- already has a ratio of 1e16,
    # and reporting that as "degenerate" would hide a real measurement behind a None.
    cond = None if (not gravity_moves(qs) or smin <= 0) else smax / smin
    return {
        "n_samples": int(A.shape[0]),
        "cond": cond,
        "sigma_min": smin,
        "sigma_max": smax,
        # how much the gravity direction actually changed, against the 9.81 that a
        # full reorientation would give
        "gravity_spread_ms2": [round(v, 4) for v in spread],
        "gravity_spread_frac": round(max(spread) / 9.81, 5),
        "identifiable": bool(cond is not None and cond < IDENTIFIABLE_COND),
    }


def gravity_moves(qs, tol: float = 1e-12) -> bool:
    """Does the gravity direction as the tool sees it vary across this recording?

    This is the physical precondition for the whole module, and it is cheap to ask
    directly instead of inferring it from a singular value. `fit_mass_scalar` uses the
    same quantity as the denominator of its estimate, so "the gravity direction moves"
    and "a mass was estimated" are the same question asked twice.
    """
    import numpy as np

    G = np.asarray(gravity_columns(qs), dtype=float)
    Gc = G - G.mean(0)
    return bool(float((Gc ** 2).sum()) > tol)


def fit_mass_bias(qs, force):
    """The unrestricted least-squares (mass, bias): a separate coefficient for every
    axis of the gravity direction. Reported whether or not the answer is meaningful --
    the point is to *see* the meaningless value rather than to have the code refuse and
    hide it.

    Units: the gravity column is R(q)^T g with |g| = 9.81 m/s^2 already inside it, so the
    matching coefficient is a mass in kilograms with no further conversion. (An earlier
    version divided by 9.81 as well and reported masses nine times too large. The
    positive-control test on synthetic data with a known 4 kg mass is what caught it.)
    """
    import numpy as np

    A = design(qs)
    F = np.asarray(force, dtype=float)
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)   # (4, 3): 3 gravity columns + bias
    pred = A @ coef
    resid = F - pred
    tot = ((F - F.mean(0)) ** 2).sum()
    M = np.asarray(coef[:3], dtype=float)
    sv = np.linalg.svd(M, compute_uv=False)
    bias = np.asarray(coef[3], dtype=float)
    # If the model were right the gravity block would be m * I -- rank one, with all
    # three rows carrying the same mass. The block is reported rather than only its
    # norm, because a block that is far from rank one *is* the diagnosis: the solve
    # spread one mass across three directions, and any scalar read off it is a number
    # without a referent.
    return {"mass_kg": round(float(np.linalg.norm(M, "fro") / math.sqrt(3)), 3),
            "mass_vector": [round(float(v), 4) for v in M[:, 0]],
            "mass_block": [[round(float(v), 3) for v in row] for row in M],
            "mass_block_rank1_ratio": round(float(sv[0] / max(sv[1], 1e-30)), 2),
            "mass_block_condition": round(float(sv[0] / max(sv[-1], 1e-30)), 2),
            "bias_N": [round(float(v), 4) for v in bias],
            "r2_force": round(float(1.0 - (resid ** 2).sum() / max(tot, 1e-30)), 4)}


def fit_mass_scalar(qs, force):
    """The mass the recording actually supports, with an error bar.

    The unrestricted fit above has three coefficients for one physical quantity, and on
    a poorly excited recording it uses that freedom to fit noise. The honest question is
    the *restricted* one: f = m * R(q)^T g + b, one shared m and one bias per axis. Under
    that model m has a sampling distribution, so the answer is not a number but a number
    and an interval -- and on a recording where the tool barely reorients the interval is
    wider than the answer, which is what "not identifiable" means when it is said
    carefully.

    The algebra is the ordinary least squares of a shared slope across three axis-wise
    regressions, obtained by centring: with Gc and Fc the mean-subtracted gravity and
    force, m_hat = <Gc, Fc> / <Gc, Gc>, the bias follows, and se(m_hat) comes from the
    residual with 3n - 4 degrees of freedom.

    The t statistic is the verdict: |t| below 2 means this recording cannot tell the
    tool's mass from zero, let alone from the sensor's bias.
    """
    import numpy as np

    G = gravity_columns(qs)
    F = np.asarray(force, dtype=float)
    n = G.shape[0]
    Gc = G - G.mean(0)
    Fc = F - F.mean(0)
    denom = float((Gc ** 2).sum())
    if denom <= 1e-18:
        return {"mass_kg": None, "stderr_kg": None, "ci95_kg": None,
                "t_stat": 0.0, "identified": False, "dof": int(3 * n - 4),
                "gravity_excitation_sum_sq": 0.0,
                "reason": "the gravity direction does not change at all in this recording"}
    m = float((Gc * Fc).sum() / denom)
    resid = Fc - m * Gc
    dof = int(3 * n - 4)
    sigma2 = float((resid ** 2).sum()) / max(dof, 1)
    se = float(np.sqrt(sigma2 / denom))
    t = m / se if se > 0 else 0.0
    r2 = 1.0 - float((resid ** 2).sum()) / max(float((Fc ** 2).sum()), 1e-30)
    return {"mass_kg": round(m, 3), "stderr_kg": round(se, 3),
            "ci95_kg": [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)],
            "t_stat": round(t, 2), "identified": bool(abs(t) >= 2.0),
            "dof": dof, "gravity_excitation_sum_sq": round(denom, 4),
            "r2_force": round(r2, 4),
            "reason": ("the mass is distinguishable from zero" if abs(t) >= 2.0 else
                       "the mass is not distinguishable from zero: the tool's mass, the "
                       "sensor bias and the contact offset are one lump in this recording")}


def excited_by(q_j) -> float:
    """How much the gravity column moves when joint `j` moves, over a wide one-sided
    range, relative to the column at the base configuration. Zero means the joint cannot
    excite gravity at all."""
    base = [0.0] * 6
    a = gravity_columns([base])[0]
    best = 0.0
    for off in (-0.6, -0.2, 0.2, 0.6):
        q = list(base)
        q[j] = off
        b = gravity_columns([q])[0]
        best = max(best, max(abs(a[k] - b[k]) for k in range(3)))
    return float(best)


def inert_joints(qs, span: float = 0.6, tol: float = 1e-9) -> list:
    """Joints whose motion cannot change the gravity direction as the tool sees it,
    at this configuration.

    Rotating about an axis parallel to gravity leaves R(q)^T g unchanged: the gravity
    vector in tool coordinates does not move. Measuring that is the one place in this
    module where the answer is a property of the *machine* rather than of the recording.

    The probe is one-sided on purpose. Comparing q_j - span against q_j + span looks
    symmetric and natural, and it is wrong: a joint whose effect on the gravity column
    is even about the recorded configuration -- the ordinary case for a revolute joint,
    where the effect goes as cos(theta) -- has its two ends cancel, and the joint is
    declared inert while it is doing useful work. Each offset is compared against the
    *unperturbed* configuration instead, which cannot cancel.
    """
    base = list(qs[0])
    a = gravity_columns([base])[0]
    out = []
    for j in range(len(base)):
        alive = 0.0
        for off in (-span, -span / 3.0, span / 3.0, span):
            q = list(base)
            q[j] += off
            b = gravity_columns([q])[0]
            alive = max(alive, max(abs(a[k] - b[k]) for k in range(len(a))))
        if alive <= tol:
            out.append(j)
    return out


def gravity_inert_joints(n_joints: int = 6, configs=None, span: float = 0.6) -> dict:
    """Structural versus situational inertness.

    `structural` is a joint that does not move the gravity direction at any sampled
    configuration -- for a UR5 that is joint 1 alone, and no amount of data collection
    can change it. `situational` is a joint that is inert at some configurations and
    useful at others. "Collect more data" is a useful reply only for the second kind;
    for the first, the recording geometry itself has to change.

    The sampled set is deterministic and covering rather than random: every joint is
    swept across its whole range while the others sit somewhere varied, so the answer
    does not depend on a seed and does not depend on a generator being able to allocate.
    """
    if configs is None:
        rnd = random.Random(0)
        configs = [[0.0] * n_joints]
        for j in range(n_joints):
            for v in (-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi):
                q = [rnd.uniform(-1.5, 1.5) for _ in range(n_joints)]
                q[j] = v
                configs.append(q)
    ever, always = set(), set(range(n_joints))
    for q in configs:
        inert = set(inert_joints([q], span=span))
        ever |= inert
        always &= inert
    return {"structural": sorted(always), "situational": sorted(ever - always),
            "n_configurations_checked": len(configs)}


def excitation_required(qs, target_cond: float = 1e3) -> dict:
    """How much more motion the recording needs for the mass/bias split to reach
    `target_cond`.

    What the ratio means, stated precisely, because the useful quantity is not the
    condition number itself. `sigma_min` of [R^T g | 1] is proportional to how much the
    gravity direction varies across the recording, while `sigma_max` is dominated by its
    constant part -- so the *standard error of the fitted mass* falls as the gravity
    direction's spread grows, and it is that spread which more motion has to increase.
    `excitation_factor_needed` is therefore reported as the square root of the
    condition-number ratio and labelled as the factor by which the gravity direction's
    variation must grow; `conditioning_factor_needed` is the raw ratio, kept because it
    is the number that can be checked directly against `condition()`.

    The condition number itself is scale invariant, which is worth saying out loud
    because it means widening a joint that is inert does not help at all -- widening
    joint 1 of a UR5 changes nothing, and this function correctly refuses to move.

    Three cases are handled explicitly rather than left to arithmetic:

      * a joint whose recorded range is zero has no range to scale, so multiplying it by
        any factor leaves zero and the advice is the cap. (The first version returned nan
        here, because a zero range times an infinite factor is nan. CI on Python 3.10
        found it and this machine did not, which is the argument for running CI on more
        than one interpreter.)
      * a recording whose gravity column is exactly constant has sigma_min of exactly
        zero and an infinite ratio; the factor is reported as None rather than as inf,
        so that `json` can write the result without emitting the non-standard token.
      * the required range is capped at one full joint revolution, because a joint cannot
        rotate further and reporting 1e29 radians would be arithmetic rather than advice.

    Whichever number is quoted, the reading is the same: this is a statement that the
    experiment must be changed, not that a better estimator is needed.
    """
    import numpy as np

    A = design(qs)
    s = np.linalg.svd(A, compute_uv=False)
    smax, smin = float(s[0]), float(s[-1])
    # Same physical test as `condition()`. A design that is under-excited but not
    # constant keeps a finite ratio, however large; only a recording in which the gravity
    # direction does not move at all has no ratio to quote.
    cur = None if (not gravity_moves(qs) or smin <= 0) else smax / smin
    factor = (cur / target_cond) if cur is not None else None
    Q = np.asarray(qs, dtype=float)
    rng = np.ptp(Q, axis=0)
    max_range = 2.0 * math.pi

    def required(v: float) -> float:
        """The advice for one joint. A joint that never moved is told to use its whole
        range, because there is no smaller amount of motion that would help."""
        if v <= 0.0 or factor is None or not math.isfinite(factor):
            return max_range
        return min(float(v) * max(1.0, factor), max_range)

    req = [required(float(v)) for v in rng]
    capped = factor is None or any(float(v) * max(1.0, factor) > max_range for v in rng)
    still = int((rng < 0.1).sum())
    inert = inert_joints(qs)
    return {
        "current_cond": cur, "target_cond": target_cond,
        "conditioning_factor_needed": round(factor, 1) if factor is not None else None,
        "excitation_factor_needed": (round(float(factor) ** 0.5, 1)
                                     if factor is not None else None),
        "current_joint_range_rad": [round(float(v), 4) for v in rng],
        "required_joint_range_rad": [round(v, 4) for v in req],
        "required_range_capped_at_full_turn": bool(capped),
        "joints_barely_moving": still,
        "joints_that_cannot_excite_gravity_at_all": inert,
        "excitable_joints": [j + 1 for j in range(len(rng)) if j not in inert],
        "reading": ("the recorded motion does not excite the gravity direction: "
                    "mass and sensor bias cannot be separated from this data"),
    }
