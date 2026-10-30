"""The Muse Wonder physicist loop: tools, budgets and orchestration (build plan Part 13).

    from ghostline.agent.stage import run_physicist, rerun_model

Modules:
    inputs     harness inputs from the trajectory, calibration and trial (the training window)
    session    the state `run_review` picks up (experiments/{id}/agent/session.json)
    lab        harness runs, held-out selection, records and notebook, for one experiment
    tools      Wonder's seven tools; schemas has their arguments
    loop       the HYPOTHESIZE -> FIT -> ANALYZE -> REVISE -> CONCLUDE state machine
    conclusion submit_conclusion checks against the harness (numbers.py), and recording
    fallback   library-only mode
    stage      the entry points jobs/stages.py calls
"""
