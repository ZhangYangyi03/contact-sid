"""Derivatives from encoder data, and the one lesson that costs a day every time.

The consequence of double-differencing 30 Hz joint encoders with plain finite
differences is described in the study's README as a trap because it is one: joint
positions are quantised, the quantisation is roughly constant in amplitude, and
differentiation amplifies high-frequency noise by the frequency squared. Two
rounds of np.gradient on this data produces accelerations that are mostly
quantisation noise, and a model fitted to those is fitted to noise.

The fix is a Savitzky-Golay local polynomial fit: it smooths and differentiates in
one step, and its whole point is that the smoothing window is chosen by the user
rather than implied by the differencing. Implemented here rather than taken from
scipy so the repository's core stays dependency-free, and so the window and the
polynomial order are visible arguments rather than defaults.

The parameters below are declared, not tuned. `bench/experiment.py` reports what
the free-motion fit does as the window changes, so a reader can see the dependence
instead of being asked to trust the choice.
"""

from __future__ import annotations

import math

DEFAULT_WINDOW = 11          # frames; 11 at 30 Hz is ~0.37 s, comfortably under
                             # the contact events whose onset the study measures
DEFAULT_POLY = 2


def _solve(A, b):
    """Least squares via normal equations, with a tiny ridge for conditioning.
    Small systems only, which is what a per-window polynomial fit is."""
    m, n = len(A), len(A[0])
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    for i in range(n):
        AtA[i][i] += 1e-12
    Atb = [sum(A[k][i] * b[k] for k in range(m)) for i in range(n)]
    # Gaussian elimination with partial pivoting
    M = [row[:] + [Atb[i]] for i, row in enumerate(AtA)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            continue
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f:
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def _pinv_rows(A):
    """Row-wise pseudo-inverse of a small matrix: M = (A^T A)^{-1} A^T, returned as
    rows so M[k] is the convolution kernel that produces polynomial coefficient k.

    This is the piece that is easy to get wrong. What is wanted is not the fitted
    coefficient vector for one window -- that would have to be recomputed for every
    window -- but the *kernel* which turns any window of samples into coefficient k.
    M[k] is that kernel, and it is obtained by solving the normal equations once per
    column of A rather than once per window.
    """
    m, n = len(A), len(A[0])
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    for i in range(n):
        AtA[i][i] += 1e-12
    rows = []
    for k in range(n):
        e = [1.0 if i == k else 0.0 for i in range(n)]
        rows.append(_solve(AtA, e))          # column of (A^T A)^{-1}
    # M[k][j] = sum_i rows[k][i] * A[j][i]
    return [[sum(rows[k][i] * A[j][i] for i in range(n)) for j in range(m)] for k in range(n)]


def savgol(y, window: int = DEFAULT_WINDOW, poly: int = DEFAULT_POLY):
    """(smoothed, first_derivative) for one signal.

    The window is fitted with a polynomial in sample index, and the same fit gives
    both outputs: coefficient 0 is the smoothed value, coefficient 1 is the slope in
    units of samples. `smooth_and_derivatives` divides by dt; this function does not,
    so it stays usable on data with no time base.

    Edge handling: the window is folded in by *linear* extension of the two samples at
    the end, so the first and last (window-1)/2 samples are extrapolated rather than
    dropped. Dropping them would delete exactly the frames around contact onset, which
    is the part under study.

    The extrapolation is linear rather than a constant repeat, and that is not a
    detail: a constant repeat presents each end as a perfectly flat signal, and a
    local polynomial fitted to a flat run returns a slope of zero -- so every edge
    sample would report that the robot had stopped, which is a fabricated event of
    exactly the kind this study is about. The first version of this function did the
    constant repeat, and the test that catches it is
    test_edge_derivative_does_not_fabricate_a_stop.
    """
    n = len(y)
    if window % 2 == 0:
        window += 1
    if window < 3 or n < window:
        return list(y), [0.0] * n
    half = window // 2
    A = [[float(t) ** k for k in range(poly + 1)] for t in range(-half, half + 1)]
    M = _pinv_rows(A)                       # (poly+1) x window
    slope_lo = (y[1] - y[0]) if n > 1 else 0.0
    slope_hi = (y[-1] - y[-2]) if n > 1 else 0.0
    ypad = ([y[0] - slope_lo * (half - i) for i in range(half)]
            + list(y)
            + [y[-1] + slope_hi * (i + 1) for i in range(half)])
    sm = [0.0] * n
    d1 = [0.0] * n
    for i in range(n):
        win = ypad[i:i + window]
        sm[i] = sum(M[0][j] * win[j] for j in range(window))
        d1[i] = sum(M[1][j] * win[j] for j in range(window)) if poly >= 1 else 0.0
    return sm, d1


def smooth_and_derivatives(arr, dt: float, window: int = DEFAULT_WINDOW, poly: int = DEFAULT_POLY):
    """(q_smoothed, dq, ddq) for an (n, m) array. Derivatives come from the same
    local polynomial as the smoothing, and dq is divided by dt because the fit above
    is in sample index, not seconds."""
    cols_sm, cols_d, cols_dd = [], [], []
    for j in range(len(arr[0])):
        y = [row[j] for row in arr]
        p2 = savgol(y, window, poly)
        # second derivative: differentiate the smoothed signal once more
        p1 = savgol(p2[0], window, poly)
        cols_sm.append(p2[0])
        cols_d.append([v / dt for v in p2[1]])
        cols_dd.append([v / (dt * dt) for v in p1[1]])
    n = len(arr)
    return ([[cols_sm[j][i] for j in range(len(cols_sm))] for i in range(n)],
            [[cols_d[j][i] for j in range(len(cols_d))] for i in range(n)],
            [[cols_dd[j][i] for j in range(len(cols_dd))] for i in range(n)])


def quantisation_step(y) -> float:
    """The smallest non-zero gap between sorted unique samples -- the encoder's
    resolution as it actually appears in the recording. Reported because the
    differentiation noise floor follows from it, and a reader can then judge whether
    the second derivative is usable at all."""
    u = sorted(set(round(v, 12) for v in y))
    if len(u) < 3:
        return 0.0
    gaps = [u[i + 1] - u[i] for i in range(len(u) - 1)]
    return min(g for g in gaps if g > 0) if any(g > 0 for g in gaps) else 0.0
