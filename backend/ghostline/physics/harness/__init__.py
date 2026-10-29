"""Code that runs inside the Modal Sandbox (build plan Part 11).

Entry point: `python -m ghostline.physics.harness.run_fit /work/job.json` reads `FitJob` JSON and
writes `result.json` (`FitResult`) and `plots/<model id>/*.png` next to it. Both shapes are in
`schema.py`. In short:

    job.json    {job_id, mode: validate|select|final|simulate, system, data, constants, models,
                 splits, primary_split, n_starts, forecast_draws, bootstrap, perturbations,
                 plots, time_limit_s, workers}
    data        {t, series: {observable: [..|null]}, sigma: {observable: s}, weights, segment_id}
    models[]    {id, library_ref} or {id, code, params, observables, state, constants} + {fixed}
    result.json {job_id, mode, models: [{id, status, error, n_physical, splits: [{split,
                 params, train diagnostics, heldout_rmse, heldout_errors, heldout_draws}], full,
                 aic, bic, bootstrap_done, replicates, plots}], validation, simulations}

`select` fits on 65%, 50% and 80% of the window and scores the rest; `final` refits on all the
data with a block bootstrap and calibration/timing refits. The trusted side (`physics/select.py`)
turns a `select` result into a winner. Wonder never receives `heldout_errors`, `heldout_draws`
or the held-out plots: `select.comparison_table` and `PlotRef.audience == "agent"` are its view.
"""
