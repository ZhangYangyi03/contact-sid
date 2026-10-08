"""Reading the robot episodes, and nothing else.

One file of real data is behind every number in this repository:
DORLR/ur5_ur5_force_sensor_test on Hugging Face -- a real UR5, four episodes of
20 seconds at 30 Hz, six arm joint angles at every frame and a six-axis force /
torque reading. It is copied into data/ur5/ so a clone can reproduce the study
without a network, and its provenance is recorded in data/ur5/PROVENANCE.md.

The loader exists as a module rather than as three lines inside the study so that
the field names are declared in exactly one place. `observation.wrench` is
[fX, fY, fZ, tX, tY, tZ] in the tool frame, which is asserted against the dataset's
own meta/info.json rather than assumed -- a study that silently read moments as
forces would produce a perfectly plausible and completely wrong result.

Standard library only. The parquet reader is the one place numpy is optional: if
pyarrow is absent this raises with an instruction rather than degrading, because
half a file is worse than none.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "ur5")

#: Field names, as the dataset declares them. Checked against meta/info.json at
#: load time; the assertion is cheap and the failure mode is expensive.
STATE_FIELDS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                "robotiq_85_left_knuckle_joint"]
WRENCH_FIELDS = ["fx", "fy", "fz", "tx", "ty", "tz"]


def meta() -> dict:
    with open(os.path.join(DATA, "info.json"), encoding="utf-8") as f:
        return json.load(f)


def check_fields() -> dict:
    """Assert the dataset's declared field names and shapes are the ones this code
    assumes. Returns the feature block so a caller can print it."""
    feats = meta()["features"]
    st = feats.get("observation.state", {})
    wr = feats.get("observation.wrench", {})
    if list(st.get("names") or []) != STATE_FIELDS:
        raise ValueError(f"observation.state names changed: {st.get('names')}")
    if list(wr.get("names") or []) != WRENCH_FIELDS:
        raise ValueError(f"observation.wrench names changed: {wr.get('names')}")
    if st.get("shape") != [7] or wr.get("shape") != [6]:
        raise ValueError(f"unexpected shapes: state {st.get('shape')} wrench {wr.get('shape')}")
    return {"state": st, "wrench": wr, "fps": meta().get("fps"),
            "episodes": meta().get("total_episodes")}


def episodes():
    """[(index, timestamp, q(6), wrench(6)), ...] -- one entry per episode.

    The gripper column is dropped here rather than downstream: the seventh state
    channel is the gripper knuckle, which is not part of the arm's kinematics, and
    leaving it in the joint vector would put a non-joint into a kinematic model.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("reading the episodes needs pyarrow: pip install pyarrow") from e
    import numpy as np

    check_fields()
    out = []
    for i in range(int(meta().get("total_episodes", 0))):
        path = os.path.join(DATA, f"ep{i}.parquet")
        if not os.path.exists(path):
            continue
        d = pq.ParquetFile(path).read(
            columns=["timestamp", "observation.state", "observation.wrench"]).to_pydict()
        ts = np.asarray(d["timestamp"], dtype=float).reshape(-1)
        q = np.asarray(d["observation.state"], dtype=float)[:, :6]
        w = np.asarray(d["observation.wrench"], dtype=float)
        out.append((i, ts, q, w))
    if not out:
        raise FileNotFoundError(f"no episodes under {DATA}")
    return out


def summary(eps=None) -> list:
    """A one-line description per episode, used by the CLI and by the tests."""
    import numpy as np

    eps = eps if eps is not None else episodes()
    rows = []
    for i, ts, q, w in eps:
        f = np.sqrt((w ** 2).sum(1))
        rows.append({"episode": i, "frames": int(len(ts)), "seconds": round(float(ts[-1]), 2),
                     "dt_median_s": round(float(np.median(np.diff(ts))), 5),
                     "force_N": {"median": round(float(np.median(f)), 2),
                                 "p90": round(float(np.percentile(f, 90)), 2),
                                 "max": round(float(f.max()), 2)},
                     "joint_range_rad": [round(float(v), 4) for v in np.ptp(q, axis=0)]})
    return rows
