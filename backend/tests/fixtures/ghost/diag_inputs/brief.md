System: pendulum.
The model you are diagnosing is P1 (Nonlinear pendulum): The restoring force is g sin(angle), so big swings take longer.
Equations: \ddot\theta = -\tfrac{g}{L}\sin\theta
Its signature: Period lengthens with amplitude; swing stays the same size.

Frozen from trial 1 (statistical standard errors): g = 9.8071 +/- 0.021. Constants: L = 0.5999. Trial 1 rms residual: 0.0287.
Trial 1: 4.55 s of motion, 274 points, position noise 0.35 mm, ranges {"x": [0.10777999930078772, 0.6682897085250827], "y": [-0.30060257391766754, -0.22604641564802685], "theta": [-0.47116326181519413, 0.501632198908972], "speed_max_m_s": 1.168004958215296}.
Trial 2: 9.55 s of motion. Only the starting state (theta0 = 0.37883; omega0 = -0.16691) was fitted, to the first 0.95 s.

Threshold tau = 10.11 mm (the larger of 3 sigma_noise = 1.04 mm and half the object's radius = 10.11 mm). Largest error 113.4 mm. The ghost first stayed outside the threshold for five frames in a row at 1.40 s, 0.45 s after the launch window ended.

Image 1 is the drift plot: the measured motion against the ghost's prediction above, the error against time below. Images 2 to 5 are frames at 0.95 s, 1.40 s, 1.90 s, 2.40 s after the start of the motion: the cyan translucent object is the ghost, the white line the real object's trail, the dotted cyan line the predicted path, and the ghost's outline is green below tau, amber up to 2 tau, red beyond.

Library models you may choose from (id, name, mechanism, what it predicts):
- P0: Small-angle SHM. The bob swings like a spring: the restoring force is proportional to the angle. Period independent of amplitude; swing stays the same size.
- P1: Nonlinear pendulum. The restoring force is g sin(angle), so big swings take longer. Period lengthens with amplitude; swing stays the same size.
- P2: Viscous damping. The nonlinear pendulum plus air or pivot drag proportional to speed. Amplitude decays exponentially, with the same fractional loss every swing.
- P3: Quadratic air drag. The nonlinear pendulum plus air drag proportional to speed squared. Amplitude decays fast when large, then slowly: loss per swing shrinks with amplitude.
- P4: Pivot friction. The nonlinear pendulum plus a constant dry-friction torque at the pivot. Amplitude decays linearly: the same absolute loss every swing, then it stops.

Custom model contract, if you must write code:
A custom model is Python source that defines these module-level functions. `p` is a dict of
floats holding every parameter in the spec and every constant the model declares. `s` is the
state vector, ordered as in `state`. Time starts at 0 at the first sample of the motion window.

    def initial_state(p): ...     # state at t = 0, a sequence of len(state) floats
    def rhs(t, s, p): ...         # ds/dt, a sequence of len(state) floats
    def observe(s): ...           # optional; observables from a state, default s[:len(observables)]
    def closed_form(t, p): ...    # optional; array of shape (len(observables), len(t)), skips rhs

Only `import math`, `import numpy as np` and `from scipy.special import ...` are allowed. No file
or network access, classes, `with` blocks, or names such as open, exec, eval, os and sys.
