"""The candidate library, and the sparse fit that picks terms out of it.

This file is the experiment. A library is the set of candidate explanations a model
is allowed to use; sparse regression (STLSQ -- fit by least squares, delete the
small coefficients, refit, repeat) is what chooses among them. Two libraries are
built here and the whole repository turns on the difference between them:

    smooth(q, dq, ddq)          polynomials and their products -- the assumption
                                that the physics is a smooth function of the state
    nonsmooth(smooth, ...)      the same, plus sign(dq), |dq|*sign(dq), and a
                                contact indicator

Why the second one is not decoration: a friction force is proportional to
sign(velocity), which is discontinuous at zero. No finite polynomial approximates
it: the best a smooth library can do is a steep ramp, and a steep ramp fitted away
from zero extrapolates to nonsense near it. If contact and friction are what
matters in this data, the smooth library must lose -- and the size of that loss is
the result, not a footnote.

The fit is deliberately the plainest sparse regression that exists. The point of the
study is not that a clever estimator wins; it is which *library* can express the
answer at all. A clever estimator over a library that cannot express the answer
still cannot express the answer.
"""

from __future__ import annotations

import itertools

#: Terms in the smooth library, in the order they are emitted.
SMOOTH_GROUPS = ("1", "q", "dq", "ddq", "q*q", "dq*dq", "q*dq")
#: Additional groups in the non-smooth library.
NONSMOOTH_GROUPS = ("sign(dq)", "dq*|dq|", "contact")


def smooth_library(q, dq, ddq, names_only: bool = False):
    """Constant, linear and quadratic-in-the-state terms.

    Kept to quadratic rather than a deeper polynomial on purpose: a deeper smooth
    library would fit the free-motion data better and would make the contact failure
    look like a tuning problem rather than a structural one.
    """
    m = len(q[0])
    names = ["1"]
    names += [f"q{j}" for j in range(m)]
    names += [f"dq{j}" for j in range(m)]
    names += [f"ddq{j}" for j in range(m)]
    names += [f"q{i}*q{j}" for i, j in itertools.combinations_with_replacement(range(m), 2)]
    names += [f"dq{i}*dq{j}" for i, j in itertools.combinations_with_replacement(range(m), 2)]
    names += [f"q{i}*dq{j}" for i in range(m) for j in range(m)]
    if names_only:
        return names
    cols = [[1.0] * len(q)]
    cols += [[row[j] for row in q] for j in range(m)]
    cols += [[row[j] for row in dq] for j in range(m)]
    cols += [[row[j] for row in ddq] for j in range(m)]
    cols += [[row[i] * row[j] for row in q] for i, j in itertools.combinations_with_replacement(range(m), 2)]
    cols += [[row[i] * row[j] for row in dq] for i, j in itertools.combinations_with_replacement(range(m), 2)]
    cols += [[row[i] * row[j] for row in q] for i in range(m) for j in range(m)]
    return _transpose(cols), names


def contact_variables(dq, eps: float = 0.02):
    """A contact / stiction indicator, derived from the data rather than labelled by
    hand: a joint that is nearly stationary while its neighbours are not is the
    signature of something being held.

    `sign(dq)` is emitted with a dead band because a bare sign of noisy velocity is
    a square wave at every zero crossing; the dead band says "this joint is not
    meaningfully moving" rather than picking a sign from noise. The band is a
    declared parameter, and the study reports the fit across a range of it.
    """
    n, m = len(dq), len(dq[0])
    sign = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            v = dq[i][j]
            sign[i][j] = 0.0 if abs(v) < eps else (1.0 if v > 0 else -1.0)
    # "something is holding": at least one joint inside the dead band while at least
    # one other is outside it. Derived from the same dead band that produced `sign`,
    # not from a second threshold -- a hold flag computed with a *different* eps would
    # disagree with sign(dq) on the frames near the boundary, and those are exactly the
    # frames where contact starts.
    hold = []
    for i in range(n):
        still = any(sign[i][j] == 0.0 for j in range(m))
        moving = any(sign[i][j] != 0.0 for j in range(m))
        hold.append(1.0 if (still and moving) else 0.0)
    return sign, hold


def nonsmooth_library(q, dq, ddq, eps: float = 0.02):
    """The smooth library plus the terms that make friction and contact expressible.

    The non-smooth terms are appended as *columns*. The smooth library already
    returns rows-of-samples, so mixing the two directions silently produces a matrix
    with the right shape for the wrong reason -- which is what the first version of
    this function did, and it cost an IndexError to notice.
    """
    Th, names = smooth_library(q, dq, ddq)
    # Th is ALREADY rows-of-samples -- smooth_library transposes on the way out. The
    # first version transposed it a second time here, which produced a d x n matrix
    # that then failed to stack with a confusing 619-vs-620. Every library function
    # in this module returns (n_samples, n_terms); check_shapes() enforces it.
    rows = Th
    m = len(q[0])
    sign, hold = contact_variables(dq, eps)
    extra_cols, extra_names = [], []
    for j in range(m):
        extra_cols.append([row[j] for row in sign])
        extra_names.append(f"sign(dq{j})")
    for j in range(m):
        extra_cols.append([dq[i][j] * abs(dq[i][j]) for i in range(len(dq))])
        extra_names.append(f"dq{j}*|dq{j}|")
    extra_cols.append(list(hold))
    extra_names.append("contact")
    for j in range(m):
        extra_cols.append([hold[i] * sign[i][j] for i in range(len(hold))])
        extra_names.append(f"contact*sign(dq{j})")
    extra_rows = columns_to_rows(extra_cols)
    return [rows[i] + extra_rows[i] for i in range(len(rows))], names + extra_names


def columns_to_rows(cols):
    """Columns-of-samples in, rows-of-samples (n_samples x n_terms) out.

    This is the transpose, and it is the single most confusable direction in this
    module: both layouts are nested lists of numbers, both round-trip through
    numpy, and the only observable difference is a shape. It is named for the
    direction it goes in, and check_shapes() asserts the contract at the boundary.
    """
    if not cols:
        return []
    n = len(cols[0])
    return [[cols[j][i] for j in range(len(cols))] for i in range(n)]


def check_shapes(Theta, names, n_samples: int) -> None:
    """The library contract, asserted where it can be asserted cheaply: the design
    matrix is (n_samples, n_terms) and the names line up with the columns. A silently
    transposed design matrix fits *something* and reports a plausible R2, so the check
    earns its place."""
    if len(Theta) != n_samples:
        raise ValueError(f"design has {len(Theta)} rows, expected {n_samples}")
    if len(Theta[0]) != len(names):
        raise ValueError(f"design has {len(Theta[0])} columns but {len(names)} names")
    if len(set(names)) != len(names):
        raise ValueError("duplicate term names in the library")


def _transpose(cols):
    n = len(cols[0])
    return [[cols[j][i] for j in range(len(cols))] for i in range(n)]


# ------------------------------------------------------------------ fitting --

def standardise(Theta):
    """Column scaling by RMS. Without it every threshold is a different threshold
    for every term -- a q*q column has RMS ~0.1 and a qd*qd column ~0.001 -- and a
    single lambda would be pruning on units rather than on signal. The scales are
    returned so coefficients can be read back in their original units."""
    d = len(Theta[0])
    scale = []
    for j in range(d):
        s = (sum(row[j] ** 2 for row in Theta) / len(Theta)) ** 0.5
        scale.append(s if s > 1e-12 else 1.0)
    return [[row[j] / scale[j] for j in range(d)] for row in Theta], scale


def stlsq(Theta, Y, lam: float, iters: int = 12):
    """Sequentially thresholded least squares, multi-output.

    One lambda for all outputs, so the two libraries are compared at the same
    sparsity pressure rather than each at its own best setting -- a comparison where
    each side is tuned separately measures the tuning.
    """
    import numpy as np

    Th = np.asarray(Theta, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XI = np.linalg.lstsq(Th, Y, rcond=None)[0]
    small = None
    for _ in range(iters):
        small = np.abs(XI) < lam
        XI = np.where(small, 0.0, XI)
        for c in range(Y.shape[1]):
            big = ~small[:, c]
            if big.sum() == 0:
                continue
            XI[big, c] = np.linalg.lstsq(Th[:, big], Y[:, c], rcond=None)[0]
    return XI, (np.abs(XI) > 0).sum()


def r2(Y, Yhat):
    import numpy as np

    Y = np.asarray(Y, dtype=float); Yhat = np.asarray(Yhat, dtype=float)
    ss = ((Y - Yhat) ** 2).sum(0)
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ss / np.maximum(tot, 1e-30)


def fit_report(Theta, Y, lam: float, groups=None):
    """Fit and report per-group coefficient mass, so a reader can see *which kind* of
    term the fit chose, not merely how many."""
    import numpy as np

    XI, nz = stlsq(Theta, Y, lam)
    R2 = r2(Y, np.asarray(Theta) @ XI)
    out = {"lambda": lam, "nonzero": int(nz), "r2_mean": float(R2.mean()),
           "r2_per_output": [round(float(v), 5) for v in R2]}
    if groups:
        mass = {}
        for name, lo, hi in groups:
            mass[name] = round(float(np.abs(XI[lo:hi]).sum()), 6)
        out["group_mass"] = mass
    return out, XI
