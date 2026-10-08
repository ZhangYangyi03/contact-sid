"""What is checked, and why each check is here rather than somewhere else.

Every test below either (a) pins a *convention* that would otherwise be silently
wrong -- a DH table, a parquet column order, a matrix orientation -- or (b) tests a
mechanism on synthetic data whose answer is known independently. Neither category is
a re-run of the study: the study's numbers are in bench/results.json and re-running
them proves only that the machine is the same machine.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from csid import data as D          # noqa: E402
from csid import identifiability as ID  # noqa: E402
from csid import kinematics as K    # noqa: E402
from csid import library as L       # noqa: E402
from csid import signals as S       # noqa: E402
from csid import study              # noqa: E402

try:
    import numpy as np
except ImportError:                 # pragma: no cover
    np = None


class TestDataContract(unittest.TestCase):
    """The dataset's own metadata, asserted rather than assumed. A reader who swaps
    in another LeRobot-format force dataset gets a clear failure here instead of a
    study that quietly read the wrong six columns."""

    def test_fields_match_metadata(self):
        m = D.check_fields()
        self.assertEqual(m["state"]["shape"], [7])
        self.assertEqual(m["wrench"]["shape"], [6])
        self.assertEqual(list(m["wrench"]["names"]), D.WRENCH_FIELDS)

    def test_every_episode_loads_with_the_declared_width(self):
        eps = D.episodes()
        self.assertEqual(len(eps), 4)
        for i, ts, q, w in eps:
            self.assertEqual(q.shape[1], 6)          # gripper dropped, 6 joints kept
            self.assertEqual(w.shape[1], 6)          # force 3 + torque 3
            self.assertEqual(len(ts), q.shape[0])
            self.assertEqual(len(ts), w.shape[0])

    def test_timebase_is_uniform(self):
        for i, ts, q, w in D.episodes():
            d = np.diff(ts)
            self.assertLess(float(np.abs(d - d.mean()).max()), 1e-4,
                            "ep%d has a non-uniform timebase" % i)

    def test_gripper_column_is_not_treated_as_a_joint(self):
        """The seventh state channel is the gripper knuckle. If it ever leaked into the
        joint vector the DH chain would be handed a non-joint and everything downstream
        would still run."""
        self.assertEqual(len(D.STATE_FIELDS), 7)
        for i, ts, q, w in D.episodes():
            self.assertEqual(q.shape[1], len(D.STATE_FIELDS) - 1)


class TestKinematicsConventions(unittest.TestCase):
    """A forward-kinematics function is either right or silently plausible. These pin
    the conventions that can be checked without a physical robot."""

    def test_homogeneous_form(self):
        T = K.fk([0.0] * 6)
        self.assertEqual(len(T), 4)
        self.assertEqual([T[3][j] for j in range(4)], [0.0, 0.0, 0.0, 1.0])

    def test_rotation_block_is_orthonormal(self):
        R = K.rotation([0.3, -0.7, 1.1, 0.2, -0.4, 0.9])
        for i in range(3):
            for j in range(3):
                dot = sum(R[k][i] * R[k][j] for k in range(3))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=9)

    def test_reach_is_bounded_by_the_kinematic_maximum(self):
        """Sampled over the full joint range. The second assertion is the one with
        content: the sampled poses must not exceed the sum of the link lengths, and they
        must actually get out past the *working* envelope often enough that the two
        numbers are not interchangeable."""
        qs = [[((i * 1.1 + j * 0.37) % (2 * math.pi)) - math.pi for j in range(6)]
              for i in range(200)]
        s = K.workspace_stats(qs)
        self.assertTrue(s["within_kinematic_reach"], s)
        self.assertLessEqual(s["tcp_radius_max_m"], K.REACH_KINEMATIC_M + 1e-6)
        self.assertGreater(s["tcp_radius_max_m"], 0.6)

    def test_the_two_reach_numbers_are_not_the_same_number(self):
        self.assertLess(K.REACH_WORKING_M, K.REACH_KINEMATIC_M)

    def test_no_rotation_can_move_the_tool_beyond_the_link_sum(self):
        """A kinematic bound that a random search must respect: fifteen hundred random
        configurations, none outside the sum of the DH offsets."""
        rng = np.random.default_rng(3)
        qs = rng.uniform(-math.pi, math.pi, size=(1500, 6)).tolist()
        s = K.workspace_stats(qs)
        self.assertLessEqual(s["tcp_radius_max_m"], K.REACH_KINEMATIC_M + 1e-6)

    def test_zero_configuration_matches_the_dh_table_by_hand(self):
        """Derived from the table, not from the code: at q=0 the only translations are
        a2 and a3 along the first frame's x, which frame 1's rotation leaves alone, so
        p_x must be exactly a2+a3. The remaining components are the wrist offsets moved
        by the 90-degree alpha of joint 1."""
        T = K.fk([0.0] * 6)
        a2, a3 = K.DH[1][0], K.DH[2][0]
        self.assertAlmostEqual(T[0][3], a2 + a3, places=9)
        # d4 + d6 land on the base y in this convention, d5 on z
        self.assertAlmostEqual(T[1][3], -(K.DH[3][1] + K.DH[5][1]), places=9)

    def test_joint_one_rotates_about_the_base_z_axis(self):
        """The strongest convention check available without hardware: joint 1 is the
        base rotation, so moving it cannot change the tool height and cannot change its
        distance from the base axis."""
        q = [0.2, -0.9, 1.0, 0.1, 0.5, -0.3]
        J = K.jacobian(q)
        p = np.asarray(K.fk_np([q])[0])
        self.assertAlmostEqual(float(J[2, 0]), 0.0, places=6)
        radial = np.array([p[0], p[1], 0.0])
        self.assertAlmostEqual(float(J[:, 0] @ radial), 0.0, places=6)

    def test_jacobian_agrees_with_the_columns_it_differentiates(self):
        """Central differences against the FK they came from -- self-consistent by
        construction, and worth asserting because a Jacobian built from a *different*
        FK is the classic silent mismatch."""
        q = [0.4, -0.5, 0.7, -0.2, 0.3, 0.1]
        h = 1e-6
        for j in range(6):
            qp, qm = list(q), list(q)
            qp[j] += h
            qm[j] -= h
            num = (np.asarray(K.fk_np([qp])[0]) - np.asarray(K.fk_np([qm])[0])) / (2 * h)
            analytic = K.jacobian(q)[:, j]
            self.assertLess(float(np.abs(num - analytic).max()), 1e-5)


class TestSignalDerivatives(unittest.TestCase):
    """The derivative estimator, tested where the answer is known."""

    def test_smoothing_of_a_polynomial_is_exact(self):
        """A degree-2 polynomial is inside a degree-2 fit's span, so the smoothed value
        must equal the sample -- but only where the fit is not dominated by the edge
        extrapolation, so the interior is checked separately from the ends."""
        y = [3.0 + 2.0 * i + 0.5 * i * i for i in range(40)]
        sm, _ = S.savgol(y, 11, 2)
        self.assertLess(max(abs(sm[i] - y[i]) for i in range(5, 35)), 1e-6)
        # with linear edge extension the ends are exact too, up to the linear trend
        # from the last two samples; a constant extension would be off by ~20 here
        self.assertLess(max(abs(sm[i] - y[i]) for i in range(40)), 1.0)

    def test_edge_derivative_does_not_fabricate_a_stop(self):
        """The regression test for a real bug, kept because the bug is invisible: a
        constant-valued edge extension presents each end as a flat run, a local
        polynomial fitted to a flat run returns slope zero, and every edge sample then
        reports that the robot had stopped. On a signal that is genuinely moving the
        edge slope must be close to the interior slope, not to zero."""
        y = [1.0 + 3.0 * i for i in range(60)]          # slope exactly 3 per sample
        _, d1 = S.savgol(y, 11, 2)
        for i in (0, 1, 2, 3, 4, 55, 56, 57, 58, 59):
            self.assertAlmostEqual(d1[i], 3.0, places=5)

    def test_derivative_of_a_polynomial_is_exact(self):
        """y = a*i + b*i^2 in sample index: slope = a + 2b*i. This is the answer the
        convolution kernel in _pinv_rows has to produce, with no time base involved."""
        a, b = 0.7, -0.03
        y = [a * i + b * i * i for i in range(60)]
        _, d1 = S.savgol(y, 11, 2)
        for i in range(10, 50):
            self.assertAlmostEqual(d1[i], a + 2 * b * i, places=5)

    def test_derivative_of_a_sine_is_close_but_not_exact(self):
        """A quadratic fit to a sine has a truncation error, and the test asserts the
        size it should have rather than pretending it is zero: ~1% at this window."""
        dt, f = 0.01, 1.0
        y = [math.sin(2 * math.pi * f * i * dt) for i in range(300)]
        _, d1 = S.savgol(y, 11, 2)
        err = max(abs(d1[i] / dt - 2 * math.pi * f * math.cos(2 * math.pi * f * i * dt))
                  for i in range(20, 280))
        self.assertLess(err, 0.15)
        self.assertGreater(err, 1e-4, "the estimator should not be suspiciously exact")

    def test_even_window_is_made_odd(self):
        """A symmetric window has to be odd; silently accepting an even one shifts the
        derivative by half a sample and biases every estimate."""
        y = [float(i) for i in range(30)]
        _, d1 = S.savgol(y, 10, 1)
        self.assertAlmostEqual(d1[15], 1.0, places=6)

    def test_short_signal_degrades_loudly_rather_than_crashing(self):
        out = S.savgol([1.0, 2.0], 11, 2)
        self.assertEqual(out[0], [1.0, 2.0])

    def test_quantisation_step_is_measurable(self):
        y = [round(math.sin(i / 10.0), 3) for i in range(100)]
        self.assertGreater(S.quantisation_step(y), 0)


class TestLibrary(unittest.TestCase):
    """The library contract and the sparse fit."""

    def _tiny(self, n=20):
        q = [[0.01 * i, 0.2, 0.3, 0.4, 0.5, 0.6] for i in range(n)]
        dq = [[0.5, -0.5, 0.0, 0.02, 0.0, 0.1] for _ in range(n)]
        ddq = [[0.0] * 6 for _ in range(n)]
        return q, dq, ddq

    def test_shape_contract_holds_for_both_libraries(self):
        q, dq, ddq = self._tiny()
        for build in (L.smooth_library, L.nonsmooth_library):
            Th, names = build(q, dq, ddq)
            L.check_shapes(Th, names, len(q))
            self.assertEqual(len(Th), len(q))
            self.assertEqual(len(Th[0]), len(names))
            self.assertEqual(len(set(names)), len(names))

    def test_the_two_libraries_share_a_prefix(self):
        """The non-smooth library must be the smooth one plus terms, otherwise the
        comparison between them is a comparison of two unrelated models."""
        q, dq, ddq = self._tiny()
        a, na = L.smooth_library(q, dq, ddq)
        b, nb = L.nonsmooth_library(q, dq, ddq)
        self.assertEqual(nb[:len(na)], na)
        self.assertGreater(len(nb), len(na))
        self.assertEqual(a, [row[:len(na)] for row in b])

    def test_mis_shaped_design_is_rejected(self):
        q, dq, ddq = self._tiny()
        Th, names = L.smooth_library(q, dq, ddq)
        with self.assertRaises(ValueError):
            L.check_shapes(L.columns_to_rows(Th), names, len(q))

    def test_columns_to_rows_is_the_transpose(self):
        q, dq, ddq = self._tiny(5)
        Th, _ = L.smooth_library(q, dq, ddq)
        self.assertEqual(L.columns_to_rows(L.columns_to_rows(Th)), Th)

    def test_sign_is_dead_banded_not_a_square_wave(self):
        """sign(dq) of noisy velocity is a square wave at every zero crossing; the dead
        band is the whole reason the term is usable, so it is asserted. The boundary
        itself is asserted too, in the direction the comparison actually takes: a value
        exactly at eps is outside the band."""
        dq = [[0.001, -0.001, 0.5, -0.5, 0.0, 0.02] for _ in range(3)]
        sign, hold = L.contact_variables(dq, eps=0.02)
        self.assertEqual(sign[0][:3], [0.0, 0.0, 1.0])
        self.assertEqual(sign[0][3], -1.0)
        self.assertEqual(sign[0][4], 0.0)
        self.assertEqual(sign[0][5], 1.0)

    def test_hold_flag_agrees_with_the_sign_it_is_derived_from(self):
        """`hold` must not use a second threshold. On a row where every joint is inside
        the band the flag is zero even though joints are still, and on a row where every
        joint is outside it the flag is zero too -- only a mix counts."""
        all_still = [[0.0, 0.0, 0.001, 0.0, 0.0, 0.0]]
        all_moving = [[0.5, -0.5, 0.6, 0.7, 0.55, 0.8]]
        mixed = [[0.5, -0.5, 0.0, 0.0, 0.0, 0.8]]
        self.assertEqual(L.contact_variables(all_still, 0.02)[1], [0.0])
        self.assertEqual(L.contact_variables(all_moving, 0.02)[1], [0.0])
        self.assertEqual(L.contact_variables(mixed, 0.02)[1], [1.0])

    def test_stlsq_recovers_a_known_sparse_system(self):
        """The mechanism, on synthetic data: a target built from three known columns of
        a 12-column library must be recovered with the other nine at zero."""
        rng = np.random.default_rng(0)
        n, d = 400, 12
        X = rng.normal(size=(n, d))
        w = np.zeros((d, 2))
        w[1, 0], w[4, 0], w[7, 1] = 2.0, -1.5, 0.75
        Y = X @ w
        XI, nz = L.stlsq(X.tolist(), Y, 0.01)
        self.assertEqual(list(np.nonzero(XI[:, 0])[0]), [1, 4])
        self.assertEqual(list(np.nonzero(XI[:, 1])[0]), [7])
        self.assertAlmostEqual(float(XI[1, 0]), 2.0, places=6)

    def test_r2_is_one_for_a_perfect_fit_and_zero_for_the_mean(self):
        Y = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertAlmostEqual(float(L.r2(Y, Y).mean()), 1.0, places=12)
        mean_only = np.tile(Y.mean(0), (3, 1))
        self.assertAlmostEqual(float(L.r2(Y, mean_only).mean()), 0.0, places=12)

    def test_standardise_can_be_undone(self):
        """Column scaling is a change of variables; if it cannot be undone then every
        coefficient printed in the study is in the wrong units."""
        X = np.asarray([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        Xs, scale = L.standardise(X.tolist())
        back = np.asarray([[Xs[i][j] * scale[j] for j in range(2)] for i in range(3)])
        self.assertLess(float(np.abs(back - X).max()), 1e-12)


class TestIdentifiability(unittest.TestCase):
    """The discrimination the whole identifiability half depends on: it must say NO on
    a degenerate recording and YES on a well-excited one. A test that only showed the
    negative case could be passed by a function that always says no."""

    def test_a_tool_that_never_reorients_cannot_separate_mass_from_bias(self):
        qs = [[0.1 * i, 0.3, 0.3, 0.3, 0.3, 0.3] for i in range(100)]
        c = ID.condition(qs)
        self.assertFalse(c["identifiable"])
        self.assertGreater(c["cond"], 1e3)

    def test_a_tool_that_reorients_widely_can(self):
        qs = []
        for i in range(200):
            t = i / 199.0
            qs.append([-1.5 + 3.0 * t, -1.2 + 2.4 * t, -1.0 + 2.0 * t,
                       -1.5 + 3.0 * t, -1.5 + 3.0 * t, -1.5 + 3.0 * t])
        c = ID.condition(qs)
        self.assertTrue(c["identifiable"], c)
        self.assertLess(c["cond"], c["cond"] + 1)

    def test_a_degenerate_recording_returns_an_absurd_mass_and_does_not_hide_it(self):
        rng = np.random.default_rng(1)
        qs = [[0.02 * rng.normal(), 0.3, 0.3, 0.3, 0.3, 0.3] for _ in range(200)]
        F = (ID.gravity_columns(qs) * 6.5 + 9.0 + 0.01 * rng.normal(size=(200, 3))).tolist()
        fit = ID.fit_mass_bias(qs, F)
        self.assertGreater(abs(fit["mass_vector"][0]) + abs(fit["mass_vector"][1]), 1e-6)
        self.assertIn("r2_force", fit)
        # and the block that would betray the split is reported with it
        self.assertEqual(len(fit["mass_block"]), 3)
        self.assertGreaterEqual(fit["mass_block_rank1_ratio"], 1.0)

    def test_a_well_conditioned_recording_recovers_a_known_mass(self):
        """The positive control for both fits, and the test that caught a factor-of-9.81
        units error in the unrestricted one: on synthetic data generated from a known
        mass of 4 kg with a known 5 N bias over a widely reorienting motion, both must
        return the mass and the bias they were given. Without this the negative result
        above could come from functions that never work."""
        rng = np.random.default_rng(11)
        n = 600
        qs = [[rng.uniform(-2.0, 2.0) for _ in range(6)] for _ in range(n)]
        G = ID.gravity_columns(qs)
        F = (G * 4.0 + 5.0).tolist()
        fit = ID.fit_mass_bias(qs, F)
        self.assertAlmostEqual(fit["mass_kg"], 4.0, places=1)
        self.assertAlmostEqual(float(fit["bias_N"][0]), 5.0, places=1)
        self.assertGreater(fit["r2_force"], 0.999)
        sc = ID.fit_mass_scalar(qs, F)
        self.assertAlmostEqual(sc["mass_kg"], 4.0, places=2)
        self.assertLess(sc["stderr_kg"], 0.05)
        self.assertTrue(sc["identified"], sc)

    def test_a_recording_that_does_not_move_identifies_nothing(self):
        """The negative control for the restricted fit, stated at its true strength. A
        tool that never changes orientation gives exactly one value of R(q)^T g for the
        whole recording, so the slope has no denominator and the fit must refuse rather
        than divide by something near zero and return a number.

        Note what this test deliberately does *not* claim. Once the orientation does
        vary at all, independent noise averages out and even a tiny excursion identifies
        the mass given enough samples -- the binding constraint is the design's
        conditioning, not the sample count. That distinction is the point of the study's
        second finding, so the test is written to hold the first claim without
        overstating the second."""
        rng = np.random.default_rng(12)
        qs = [[0.0, 0.3, 0.25, 0.2, 0.15, 0.2] for _ in range(400)]
        G = ID.gravity_columns(qs)
        F = (G * 8.0 + 20.0 + rng.normal(size=(400, 3))).tolist()
        sc = ID.fit_mass_scalar(qs, F)
        self.assertIsNone(sc["mass_kg"])
        self.assertFalse(sc["identified"], sc)
        self.assertIn("does not change at all", sc["reason"])

    def test_the_excitation_requirement_is_not_below_one_and_is_per_configuration(self):
        """Two things stated instead of assumed. The condition-number ratio is scale
        invariant, so two recordings that differ only in the size of the joint range of
        an *inert* joint must come out identical -- asserting a decrease there was the
        first version of this test, and it was wrong. And the factor is never below one:
        "you already have enough" is not an excuse to ask for a sub-unit increase."""
        a = ID.excitation_required([[0.01, 0.2, 0.25, 0.2, 0.15, 0.2] for _ in range(50)])
        b = ID.excitation_required([[1.4, 0.2, 0.25, 0.2, 0.15, 0.2] for _ in range(50)])
        # joint 1 widened by 100x, the rest identical: the gravity direction is untouched
        self.assertAlmostEqual(a["conditioning_factor_needed"],
                               b["conditioning_factor_needed"], places=6)
        self.assertGreaterEqual(a["excitation_factor_needed"], 1.0)
        self.assertGreaterEqual(a["excitation_factor_needed"],
                                a["conditioning_factor_needed"] ** 0.5 - 1e-6)
        self.assertNotIn(1, a["excitable_joints"])

    def test_a_widely_reorienting_tool_asks_for_less_motion_than_a_stiff_one(self):
        """The direction that must hold: on a recording whose gravity direction already
        swings widely, the factor asked for is smaller than on one where the tool keeps
        its orientation. Here the difference is made by the *useful* joints."""
        stiff = ID.excitation_required([[0.0] + [0.2, 0.25, 0.2, 0.15, 0.2] * 1 for _ in range(50)])
        swing = ID.excitation_required([[0.0, 0.2 + 1.2 * (i % 2), 0.25, 0.2, 0.15, 0.2]
                                        for i in range(50)])
        self.assertLess(swing["conditioning_factor_needed"], stiff["conditioning_factor_needed"])

    def test_required_range_is_capped_at_a_full_revolution(self):
        """A joint cannot turn more than once; when the deficit is large enough the
        advice must be "move it through its full range", not 1e29 radians."""
        r = ID.excitation_required([[1e-4, 0.3, 0.3, 0.3, 0.3, 0.3] for _ in range(50)])
        for v in r["required_joint_range_rad"]:
            self.assertLessEqual(v, 2 * math.pi + 1e-9)

    def test_joint_one_of_the_ur5_cannot_excite_the_gravity_direction(self):
        """The structural fact behind the negative result, asserted so that a change to
        the DH table cannot quietly invalidate the explanation: joint 1 rotates about
        the vertical, gravity is vertical, so rotating the whole arm about that axis
        leaves the gravity direction in tool coordinates exactly unchanged."""
        inert = ID.inert_joints([[0.0] * 6])
        self.assertIn(0, inert)
        for j in (1, 2, 3, 5):
            lo, hi = [0.0] * 6, [0.0] * 6
            lo[j], hi[j] = -0.7, 0.7
            a = ID.gravity_columns([lo])[0]
            b = ID.gravity_columns([hi])[0]
            self.assertGreater(max(abs(a[k] - b[k]) for k in range(3)), 1.0)

    def test_the_inertness_probe_cannot_be_fooled_by_an_even_response(self):
        """The regression test for a measurement bug that produced a confident wrong
        answer. A joint whose effect on the gravity column is even about the recorded
        configuration -- the usual case for a revolute joint, since the effect goes as
        cos(theta) -- looks inert under a symmetric +/- span probe, because the two ends
        cancel. The one-sided probe must call that joint useful.

        Checked on the arm at a configuration where joint 5's effect is even: the joint
        is useful, and would have been declared structurally inert by the symmetric
        version."""
        q = [0.4, -0.9, 1.0, 0.1, 0.5, -0.3]
        a = ID.gravity_columns([q])[0]
        for off in (-0.6, 0.6):
            b = ID.gravity_columns([[q[0], q[1], q[2], q[3], q[4] + off, q[5]]])[0]
            self.assertGreater(max(abs(a[k] - b[k]) for k in range(3)), 0.1)
        self.assertNotIn(4, ID.inert_joints([q]))

    def test_structural_inertness_is_separated_from_situational(self):
        """Joint 1 is inert everywhere; joints 2-6 sometimes line up with gravity and
        sometimes do not. The distinction is what makes "collect more data" a useful
        or a useless reply, so it is asserted rather than left to the prose."""
        d = ID.gravity_inert_joints(6)
        self.assertIn(0, d["structural"])
        for j in (1, 2, 3, 5):
            self.assertNotIn(j, d["structural"])
        self.assertTrue(set(d["structural"]).isdisjoint(d["situational"]))
        self.assertLessEqual(max(d["structural"] + d["situational"] + [0]), 5)

    def test_real_recording_is_reported_unidentifiable(self):
        """The study's own claim, asserted: the four real episodes do not excite the
        gravity direction enough to separate mass from bias."""
        eps = D.episodes()
        qs = [q.tolist() for _, _, q, _ in eps]
        q = [row for e in qs for row in e]
        c = ID.condition(q)
        self.assertFalse(c["identifiable"], c)


class TestStudyMechanics(unittest.TestCase):
    """The split machinery, which is where a study like this is most easily fooled."""

    def test_the_two_regime_masks_are_disjoint(self):
        f = np.linspace(0.0, 100.0, 1000)
        sp = study.regime_split(f)
        self.assertFalse(bool((sp["free"] & sp["contact"]).any()))
        self.assertLess(sp["free_max_N"], sp["contact_min_N"])

    def test_held_out_episode_never_appears_in_training_rows(self):
        """The leave-one-episode-out split, checked directly: a random row split would
        pass every other test in this file and still be worthless on 30 Hz data."""
        ed = study.load_arrays()
        Th, Y, grp, names = study.build_design(ed, "smooth")
        for k in range(len(ed)):
            tr, te = grp != k, grp == k
            self.assertEqual(int((tr & te).sum()), 0)
            self.assertEqual(int(tr.sum() + te.sum()), len(grp))

    def test_design_rows_match_the_concatenated_target(self):
        ed = study.load_arrays()
        for lib in ("smooth", "nonsmooth", "history"):
            Th, Y, grp, names = study.build_design(ed, lib)
            L.check_shapes(Th, names, len(Y))
            self.assertEqual(Th.shape[0], sum(len(e["q"]) for e in ed))

    def test_the_history_library_actually_contains_the_force_it_uses(self):
        """The diagnostic library is only meaningful if the lagged force columns are
        really the measured force, shifted. Reconstruct column 1 and compare."""
        ed = study.load_arrays()
        e = ed[0]
        Th, names = study.history_library(e["q"].tolist(), e["dq"].tolist(),
                                          e["ddq"].tolist(), e["wrench"])
        j = names.index("fz_lag1")
        col = np.asarray([row[j] for row in Th])
        self.assertAlmostEqual(float(col[0]), float(e["wrench"][0][2]), places=6)
        self.assertAlmostEqual(float(col[5]), float(e["wrench"][4][2]), places=6)
        self.assertEqual(len(names), len(Th[0]))

    def test_results_file_if_present_is_self_consistent(self):
        p = os.path.join(os.path.dirname(HERE), "bench", "results.json")
        if not os.path.exists(p):
            self.skipTest("bench/results.json not built yet")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        for k in ("data", "identifiability", "library_comparison", "smoothing_sensitivity"):
            self.assertIn(k, r)
        self.assertEqual(len(r["data"]), 4)
        for lib, rows in r["library_comparison"].items():
            self.assertEqual(len(rows), 4, lib)


if __name__ == "__main__":
    unittest.main(verbosity=2)
