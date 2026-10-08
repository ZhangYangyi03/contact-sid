"""contact-sid -- learning contact physics from a robot's own force data.

The claim this package tests: "you do not need a hand-written physics model; let
the machine fit one from a few operations." The two halves of the answer both live
here.

  library.py        which explanations are expressible at all -- a smooth library
                    cannot represent friction, because friction is sign(v)
  identifiability.py whether the recording can support the parameters being
                    fitted -- tool mass and sensor bias are separable only if the
                    tool reorients

Both are properties of the experiment rather than of the estimator, which is the
finding: a better fit does not rescue a library that cannot express the answer, and
a better estimator does not rescue a recording that does not distinguish the
parameters.
"""

__version__ = "0.1.0"

__all__ = ["data", "kinematics", "signals", "library", "identifiability", "study"]
