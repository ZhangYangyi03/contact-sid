# Practice teaches the policy, not the parameters: two ways a force recording fails to support the model it is fitted with

Draft, 2026-10. Every number below is from `bench/results.json`, produced by
`python -m csid all` on the four public episodes this repository stages. Numbers are
reported with the sample size and the split they came from.

## Abstract

A robot that performs a task a few times should be able to learn the physics behind it
without a hand-written model. We test that claim on real hardware: four 20-second
episodes from a UR5 with a wrist force/torque sensor, 2401 frames at 30 Hz. Two
results, both negative for the strong form of the claim and both specific about what
would fix it.

First, the parameters a practitioner actually wants are not in the recording. Fitting
`f = m R(q)^T g + b` -- which should recover the tool's mass -- returns
m = 1.07 +/- 1.37 kg with a 95% interval of [-1.61, 3.74] and t = 0.78. The gravity
direction as seen by the tool varies by only 1.6-6.2% of 9.81 across the recording, and
one joint of this arm cannot vary it at all: joint 1 rotates about the vertical and
gravity is vertical, so rotating the whole arm leaves `R(q)^T g` exactly unchanged at
every configuration. This is a property of the experiment, not of the estimator, and
it makes "collect more operations" the wrong remedy. We give the remedy in units a lab
can budget against: the gravity direction's variation must grow by a factor of 6.4,
and we name the joints where motion would achieve that.

Second, sparse regression over a smooth candidate library predicts the free-motion
wrench but does not extrapolate into contact, and making the library non-smooth does
not fix it. Adding `sign(dq)`, `|dq|sign(dq)` and contact-interaction terms leaves the
held-out contact fit unchanged or worse. A diagnostic arm that adds the measured axial
force's own recent history to the input list reaches R2 0.90 or better on the held-out
axial force, which locates the gap: it is a missing *input*, not the wrong functional
form. No estimator over a motion-only library closes it, because a contact normal force
is not a function of pose and joint velocity.

We report both as measured, including the respect in which the study design -- framed
as smooth versus non-smooth -- turned out to be measuring the second-order axis.

## 1. Setting

The data are `DORLR/ur5_ur5_force_sensor_test` on Hugging Face: real UR5 hardware,
LeRobot v2 format, four episodes of 20 s at 30 Hz (600-601 frames each), with seven
state channels (six arm joints plus the gripper knuckle) and a six-axis wrench
(fx..tz). The episodes are one task family: the force baselines differ between them
(median 16.3 N, 19.3 N, 35.3 N, 35.6 N), which matters for the split and is why one of
the two splits we report is computed *within* each episode.

Three modelling choices are declared rather than tuned:

- **Derivatives.** Joint velocities and accelerations come from a Savitzky-Golay local
  polynomial fit, not from repeated differencing. Double-differencing a quantised
  encoder signal at 30 Hz produces accelerations that are mostly quantisation noise.
  The window (11 frames, polynomial order 2) is declared, and `python -m csid
  smoothness` prints the dependence across windows from 5 to 31 so the reader can judge
  it rather than take it on trust.
- **Libraries.** The smooth library is 97 terms: a constant, joint positions,
  velocities, accelerations, and all quadratics among them -- a polynomial surrogate for
  the rigid-body dynamics, kept at degree two deliberately, so that a better free-motion
  fit cannot be mistaken for a fix to the contact problem. The non-smooth library is
  those 97 plus 19: `sign(dq)` per joint with a dead band, `|dq|sign(dq)` per joint, a
  contact indicator, and the indicator's interactions with the signs.
- **Estimator.** Sequentially thresholded least squares (fit, delete small coefficients,
  refit), with one threshold for all three force axes and the same threshold for both
  libraries. A comparison in which each side is tuned separately measures the tuning.

## 2. What the recording can identify

Under `f = m R(q)^T g + b`, the mass and the bias are separately identified only if the
gravity column varies independently of the constant column. Two fits are reported.

The unrestricted fit assigns a coefficient to each axis of the gravity direction --
three parameters for one physical quantity -- and returns masses of 3795, 7934, 41349
and 363 kg on the four episodes, i.e. three orders of magnitude above a real tool. This
is not an arithmetic error; the gravity block it produces is far from rank one, which is
the diagnosis. We report it because a reader who has run this fit has seen these numbers
and deserves an explanation rather than a scrub.

The restricted fit shares one mass across the three axes and carries an error bar. Over
all 2401 frames: m = 1.07 kg, se = 1.37 kg, 95% CI [-1.61, 3.74], t = 0.78, dof = 7199.
The interval contains zero. Per episode, three of four have |t| < 2 as well.

The mechanism is measurable and is not about sample size:

    gravity direction's variation, pooled      4.9% of 9.81
    condition number of [R^T g | 1]            4.2e4
    is it identifiable at 1e3?                 no
    variation would need to grow by            6.4x
    joints that cannot excite gravity, ever    [1]
    joints inert at some configurations        [5]

The last two lines are the transferable part. Joint 1 of a UR5 rotates about the
vertical, and the gravity direction in tool coordinates is invariant under rotation
about an axis parallel to gravity: the joint is inert *at every configuration*, so no
quantity of additional operations that use it can separate mass from bias. Other joints
can be inert *at a particular pose* -- a wrist that happens to line up with gravity --
which is a different and fixable problem. Separating the two turns "collect more data"
into "change the motion this way", and the required factor is a number.

## 3. What the libraries can express

Within each episode, both libraries are fitted on the frames below that episode's own
90th force percentile and evaluated on the frames at or above it (540 training, 60
held-out frames per episode). Training and test come from the same motion, the same
payload and the same session, so the episodes' different force baselines cannot
explain a failure.

    library      ep   R2 free   R2 contact (held out)   R2 axial force
    smooth        0    +0.734              +0.525            +0.657
    smooth        1    +0.730              -4.751            +0.937
    smooth        2    +0.975              +0.476            +0.669
    smooth        3    +0.530              -0.410            +0.637
    nonsmooth     0    +0.752              +0.496            +0.645
    nonsmooth     1    +0.742             -20.62             -1.244
    nonsmooth     2    +0.982              +0.224            +0.146
    nonsmooth     3    +0.558              -0.420            +0.665

The non-smooth library improves interpolation slightly and is worse on the held-out
contact frames in two of four episodes, including the episode with the largest contact
force by a wide margin (episode 1, threshold 75.05 N). Adding the terms that make
Coulomb friction expressible does not recover contact.

The diagnostic arm explains why. Appending the measured axial force's own last five
samples to the same smooth library:

    history       0    +0.741              +0.557            +0.667
    history       1    +0.738              -4.473            +0.941
    history       2    +0.981              +0.514            +0.920
    history       3    +0.729              -0.220            +0.899

On the axial direction -- the direction contact acts in -- the held-out fit reaches
0.90 or better in three of four episodes, against 0.94 / 0.64 / 0.90 / 0.64 for the same
library without the force history, and against 0.94 / -1.24 / 0.15 / 0.67 for the
non-smooth library.

This arm is not a model: it uses the sensor it is predicting, and a real deployment
cannot. Its role is to bound what is present in the signal, and the bound it gives is
that the missing quantity is *information*, not flexibility. The measured force carries
the contact state; the pose and joint velocities do not. That is a statement about the
input list, and it is why the study's original framing -- smooth library versus
non-smooth library -- was aimed at the wrong axis.

## 4. What would fix each failure

For identifiability, the remedy is kinematic: make the tool rotate with respect to
gravity. Concretely, on this arm, use joints 2, 3, 4, 5 and 6 and extend their ranges --
the current ranges are 0.50, 0.62, 0.53, 0.08 and 0.28 rad, and one of them (joint 5,
0.08 rad) is effectively stationary, which is a second, independent reason the
recording is under-excited. The required factor on the gravity direction's variation is
6.4, which the required joint ranges meet or exceed (they already reach the cap of a
full turn). `csid.identifiability.excitation_required` computes this from the recording
itself, so the advice is derivable before the experiment rather than after the failure.

For contact, the remedy is instrumental: the force channel has to be an input to
whatever predicts contact, not only its target. A hand-written rigid-body model plus a
force observer, or a learned model whose input includes the measured wrench, will do;
a motion-only library will not, whatever its terms.

## 5. Limitations

- One arm, one sensor, one task family, four episodes. The identifiability mechanism is
  a property of `R(q)^T g` and generalises; the specific numbers do not.
- `R2` in section 3 is on held-out frames from the same motions. The
  leave-one-episode-out table in `bench/results.json` is much worse for every library
  and is reported for exactly that reason: a model fitted on one operation does not
  transfer to another, in either library.
- The history arm's lag window (5 frames, 167 ms) is declared, not tuned.
- No force is being estimated here and no controller is evaluated. The claim is about
  what is identifiable from a recording, and the evidence is a table of fits.
- The contact/non-contact split is by measured force percentile, which is a proxy for
  contact rather than an observation of it. A labelled contact signal would be a
  better instrument and is not in this dataset.

## 6. Reproducibility

`pip install numpy pyarrow`, then `python -m csid all` (about three seconds), then
`python tests/test_reproducible.py`, which fails if any published number or either
headline claim has drifted. CI runs the unit tests on six Python/OS combinations and
installs the built wheel into an empty virtual environment. The data are staged in
`data/ur5/` with per-file SHA-256 in `data/ur5/PROVENANCE.md`.

The five bugs found while building this are each reproduced by a named test, and one of
them is worth singling out here because it is a measurement error rather than a coding
error: the first version of the joint-inertness probe compared `q - span` against
`q + span`, which cancels for any joint whose effect on the gravity column is even about
the recorded configuration -- the ordinary case for a revolute joint -- and so declared
a working wrist joint structurally incapable of exciting gravity. It was caught by a
test asserting that a joint *can* work, not that a joint cannot.
