"""CIFT activation-probe lab.

An offline research lab that reproduces the paper's CIFT method (Caught in the
Act(ivation), arXiv:2606.04141) on a small local model: capture readout-position
activations, fit an unsupervised Mahalanobis detector, and contrast
activation-level detection against the existing text scanner under an encoding
requested in-prompt.

This package is independent of the FastAPI ``app`` and depends on the opt-in
``cift`` dependency group (torch, transformers, scikit-learn, matplotlib). Pure
data/maths modules (``corpus``, ``detector``) import without that group;
``extraction`` and the lab entrypoint require it.
"""
