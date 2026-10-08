# contact-sid

Learning contact physics from a robot's own force data -- and finding out which half
of that promise holds.

The claim being tested is a popular one: *you do not need a hand-written physics
model; let the machine fit one from a few operations.* This repository fits one, on
real data from a real UR5, and reports that the claim is true for what a robot should
do and false for what a robot is. The distinction is measured, not asserted.

Everything here runs on one file of public data, with numpy and pyarrow as the only
dependencies, in about three seconds, on a laptop CPU. The measurements are in
`bench/results.json`; this file explains what they mean.

## The two findings

**1. Practice teaches the policy. It does not teach the parameters.**

Four episodes of a UR5 with a force/torque sensor on the wrist (DORLR/ur5, 20 s each
at 30 Hz, 2401 frames). The obvious modelling target is the tool's mass and the
sensor's bias, because those are the physics. Asked directly, the recording answers:

    m = 1.07 +- 1.37 kg     95% CI [-1.61, 3.74]     t = 0.78     identified: no

The interval contains zero, so this data cannot tell a real tool from a massless one.
The reason is one line of linear algebra. The sensor reads

    f = m * R(q)^T g + b + (everything else)

and `m` separates from `b` only if the gravity direction *as the tool sees it* varies
across the recording. It barely does: the column's spread is 1.6-6.2% of 9.81, and the
condition number of `[R^T g | 1]` is 4.2e4 against the 1e3 that would make the split
honest. Worse, it is not a sampling problem. Joint 1 of a UR5 rotates about the
vertical, and gravity is vertical, so rotating the whole arm about that axis leaves
`R(q)^T g` *exactly* unchanged -- at every configuration, forever. No number of
additional operations performed on that axis can help, and this code says so in joint
numbers rather than in prose:

    joints that cannot excite gravity at any configuration: [1]
    joints that happen to be inert at some configurations: [5]
    joints where more motion would help: [2, 3, 4, 5, 6]
    gravity direction's variation would need to grow: 6.4x
    (the required joint range is capped at a full turn, which it already reaches)

So the honest answer to "how many operations are enough" is: **operations are the
wrong unit.** What has to change is the motion, and that is budgetable before the
robot is switched on.

**2. A smooth library cannot learn contact -- and making it non-smooth does not fix
it, because friction is not the real problem.**

The smooth library is 97 terms: constant, joint positions, velocities, accelerations,
and their quadratics (a polynomial surrogate for the rigid-body dynamics). The
non-smooth library is that plus 19 more: `sign(dq)`, `|dq|sign(dq)` and a contact
indicator with its interaction terms -- the terms that make Coulomb friction and
stiction expressible at all.

Within each episode, both libraries are fitted to the frames below that episode's own
90th force percentile and then asked to predict the frames above it. Same motion, same
tool, so the episodes' different force baselines cannot explain anything.

    library        ep     R2_free   R2_contact(held out)   R2_axial_fz
    smooth          0      +0.734              +0.525           +0.657
    smooth          1      +0.730              -4.751           +0.937
    smooth          2      +0.975              +0.476           +0.669
    smooth          3      +0.530              -0.410           +0.637
    nonsmooth       0      +0.752              +0.496           +0.645
    nonsmooth       1      +0.742             -20.62            -1.244
    nonsmooth       2      +0.982              +0.224           +0.146
    nonsmooth       3      +0.558              -0.420           +0.665
    history         0      +0.741              +0.557           +0.667
    history         1      +0.738              -4.473           +0.941
    history         2      +0.981              +0.514           +0.920
    history         3      +0.729              -0.220           +0.899

The non-smooth library does not win. It is slightly better where the model is already
interpolating and *worse* on the held-out contact frames in two of four episodes. Adding
the friction terms is not the fix, which was not the expected result and is the more
interesting one.

The `history` arm is the control that explains why. It is the same smooth library with
the measured axial force's own last five samples appended to the input list -- not a
deployable model, since it uses the sensor being predicted, but an upper bound on what
is present in the signal. It lifts the held-out contact fit to R2 0.667 / 0.941 / 0.920 /
0.899, and it lifts the *axial* force -- the direction contact acts in -- to 0.90 or
better in three of four episodes even when the full six-axis fit stays poor in the
episode where the contact is largest.

That is the finding, stated without dressing: the gap is a **missing input, not a wrong
functional form**. A contact normal force is not a function of the robot's pose and
joint velocities; no estimator over a motion-only library closes that gap, and the
library's shape is a second-order question next to the input list. Reducing the
question to "smooth versus non-smooth" -- which is how this study was designed -- turns
out to be measuring the wrong axis.

## What is here

    csid/data.py             four episodes, field names asserted against the dataset's
                             own metadata rather than assumed
    csid/kinematics.py       UR5 forward kinematics and a numerical Jacobian, with the
                             DH conventions pinned by tests that do not need a robot
    csid/signals.py          Savitzky-Golay smoothing and derivatives; the trap it
                             avoids costs half a day every time
    csid/library.py          the three candidate libraries and sparse regression (STLSQ)
    csid/identifiability.py  whether the recording can support the parameters at all
    csid/study.py            the splits, and every number that gets published
    bench/results.json       the output, including the findings next to the numbers

## Running it

    pip install numpy pyarrow
    python -m csid doctor            the data, the kinematics, whether they agree
    python -m csid identifiability   can this recording support the parameters at all
    python -m csid library           smooth vs non-smooth, fitted and crossed over
    python -m csid contact           the within-episode extrapolation table
    python -m csid smoothness        how much the answer depends on the smoothing window
    python -m csid all               everything, written to bench/results.json
    python -m unittest discover -s tests

48 tests. They fall into three groups and none of them is a re-run of the study: tests
that pin a *convention* which would otherwise be silently wrong (the DH table, the
parquet column order, the orientation of the design matrix), tests of a mechanism on
synthetic data whose answer is known independently (a known 4 kg mass recovered, a
known sparse system recovered), and regression tests for the six real bugs this code
had, each of which produced a plausible number rather than an error:

  - a Savitzky-Golay kernel built from the wrong normal equations, which smoothed
    beautifully and returned a slope of the wrong shape
  - a constant-valued edge extension, which fabricated a full stop at both ends of
    every trajectory, since a local polynomial fitted to a flat run returns zero slope
  - a library function that transposed its output, so the design matrix was
    `terms x samples` and still fitted *something*, with an R2
  - a mass reported nine times too large because the gravity column already carries
    the 9.81 and the code divided by it again
  - an inertness probe comparing `q - span` against `q + span`, which cancels for any
    joint whose effect is even about the recorded configuration and declared a
    perfectly good wrist joint structurally unable to excite gravity
  - a degenerate recording (the gravity column exactly constant) that produced a `nan`
    recommendation, because a zero joint range times an infinite factor is `nan` -- found
    by CI on Python 3.10 and reproduced by neither 3.12 nor this host

The inertness probe is the most instructive: it produced a confident, specific, wrong answer
about the machine, and it was caught only by a test that asserted a joint *can* work
rather than that a joint *cannot*.

## What is deliberately not claimed

- No contact force is being estimated. The sensor's own history is used as a
  diagnostic, and using a sensor to predict itself is not a model.
- The four episodes are one task family on one arm. The identifiability result is about
  this recording's geometry; the mechanism is general, the numbers are not.
- `R2` here is on held-out frames from the same motions. The leave-one-episode-out table
  in `bench/results.json` is much worse for every library, and is reported for that
  reason.
- The smoothing window is a declared choice, not a tuned one. `python -m csid
  smoothness` prints the dependence across windows from 5 to 31 so a reader can judge
  it; the free-motion fit moves by a few hundredths and the contact failure does not
  move at all.

## Provenance

Data: `DORLR/ur5_ur5_force_sensor_test`, real UR5 hardware, LeRobot format. Staged into
`data/ur5/` so a clone reproduces without a network; see `data/ur5/PROVENANCE.md`.
The UR5 DH parameters are the published table, and the tests check them against the
published working envelope and against the kinematic bound implied by the link lengths.

MIT licensed.

## Why this is worth keeping

The same question -- *is this conclusion something the experiment can support, or only
something the fitter produced?* -- is the one thing this repository does, and it is
asked here in a domain where the answer is checkable rather than rhetorical. The
transferable part is the shape of the check: name the parameter, exhibit the column
that identifies it, measure that column's excitation, and state what would have to
change in the *experiment* before more data would help. Three of the functions in
`csid/identifiability.py` exist so that this is a computation rather than a paragraph,
and the one-number version of it (`excitation_required`) is the version a lab can act
on.
