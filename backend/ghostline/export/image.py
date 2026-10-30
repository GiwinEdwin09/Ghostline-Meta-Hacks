"""The Modal image for the PyBullet replay check (build plan Parts 11 and 19).

It is the physics fit image plus PyBullet, defined here so `physics/sandbox.py` stays as it is.
PyBullet ships no wheels, so the image build compiles it: a compiler goes in first and the
build is cached by Modal after the first run. Nothing in the image reaches the network at run
time; the Sandbox that uses it blocks it.
"""

from __future__ import annotations

from ghostline.physics.sandbox import fit_image

from .expectations import PYBULLET_VERSION

replay_image = fit_image.apt_install("build-essential").pip_install(f"pybullet=={PYBULLET_VERSION}")
