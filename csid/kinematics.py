"""Forward kinematics and the Jacobian for the UR5, from the standard DH table.

Why the kinematics matter at all: the force sensor reports in the *tool* frame and
the robot's motion is in *joint* space, so "force against displacement" -- the
relation the whole study is about -- only exists once both sides are expressed in
the same frame. The Jacobian is the piece that converts joint velocity into the
tool's Cartesian velocity, so it is the piece that decides whether the study is
measuring contact at all or just measuring encoder noise.

Written out rather than taken from a robotics library, for the same reason the
parquet loader declares its field names: the DH parameters are a claim about the
machine, and a claim sitting in a file that can be read is a claim that can be
checked. `python -m csid doctor` checks them against the reachable workspace.

Values are the UR5's published DH table (d1, a2, a3, d4, d5, d6 in metres).
"""

from __future__ import annotations

import math

DH = [
    # (a, d, alpha)
    (0.0,     0.089159,  math.pi / 2),
    (-0.425,  0.0,       0.0),
    (-0.39225, 0.0,      0.0),
    (0.0,     0.10915,   math.pi / 2),
    (0.0,     0.09465,  -math.pi / 2),
    (0.0,     0.0823,    0.0),
]

#: Two different reaches, and conflating them is an easy way to write a test that
#: fails for the wrong reason. The published 0.85 m is the *working* envelope -- what
#: UR advertises as the useful workspace, with the wrist inside its joint limits. The
#: kinematic maximum from the DH table above is what the chain can describe at all,
#: with no regard for joint limits: the forearm and upper arm in series plus the wrist
#: offsets. A pose at 0.9 m is kinematically fine and outside the working envelope,
#: which is a statement about the arm's specification rather than about the arithmetic.
REACH_WORKING_M = 0.85
REACH_KINEMATIC_M = (abs(DH[1][0]) + abs(DH[2][0])
                     + abs(DH[3][1]) + abs(DH[4][1]) + abs(DH[5][1]))


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def fk(q):
    """4x4 tool pose. Pure Python, no numpy: the tests call it on single poses."""
    T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for qi, (a, d, al) in zip(q, DH):
        ct, st = math.cos(qi), math.sin(qi)
        ca, sa = math.cos(al), math.sin(al)
        A = [[ct, -st * ca, st * sa, a * ct],
             [st, ct * ca, -ct * sa, a * st],
             [0.0, sa, ca, d],
             [0.0, 0.0, 0.0, 1.0]]
        T = _matmul(T, A)
    return T


def fk_np(qs):
    """Stack of tool positions for many configurations. numpy if available, else a
    list of tuples -- the study needs the whole trajectory, the tests need a few."""
    try:
        import numpy as np
    except ImportError:
        return [tuple(fk(q)[i][3] for i in range(3)) for q in qs]
    out = []
    for q in qs:
        T = fk(q)
        out.append([T[0][3], T[1][3], T[2][3]])
    return np.asarray(out)


def rotation(q):
    """3x3 tool rotation, for expressing the world gravity vector in the tool frame."""
    T = fk(q)
    return [[T[i][j] for j in range(3)] for i in range(3)]


def jacobian(q, h: float = 1e-6):
    """Position Jacobian by central differences. Numerical rather than analytic on
    purpose: the analytic UR5 Jacobian is easy to get subtly wrong, and a wrong one
    looks exactly like a right one until something else contradicts it. Central
    differences on the FK above are checkable against the FK itself."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("the Jacobian needs numpy: pip install numpy")
    base = fk_np([list(q)])[0]
    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        qp = list(q); qm = list(q)
        qp[i] += h; qm[i] -= h
        J[:, i] = (fk_np([qp])[0] - fk_np([qm])[0]) / (2.0 * h)
    return J


def workspace_stats(qs):
    """How far the tool travels, in metres, per Cartesian axis. This is the number
    that decides whether the episodes contain any excitation at all -- a tool that
    barely moves pinches the gravity term and the sensor bias into the same
    direction and makes them inseparable."""
    import numpy as np

    P = np.asarray(fk_np(qs), dtype=float)
    r = np.sqrt((P ** 2).sum(1))
    return {"tcp_travel_m": [round(float(v), 4) for v in np.ptp(P, axis=0)],
            "tcp_radius_min_m": round(float(r.min()), 4),
            "tcp_radius_max_m": round(float(r.max()), 4),
            "within_kinematic_reach": bool(r.max() <= REACH_KINEMATIC_M + 1e-6),
            "within_working_envelope": bool(r.max() <= REACH_WORKING_M + 1e-6)}
