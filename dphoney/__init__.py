"""DP-HONEY: calibrated honeytoken generation and detection.

Model-free (numpy/scikit-learn, no torch) implementation of the paper's second
pillar: a differentially-private character-bigram canary generator, a
distinguisher battery that measures how separable the canaries are from
format-valid reference credentials, and split-conformal calibration of the
detector. The generation path is numpy-only so the live injection path can reuse
it without the opt-in ``dphoney`` group.
"""
