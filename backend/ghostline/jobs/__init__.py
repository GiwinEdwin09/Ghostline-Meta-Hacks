"""Modal job entry points: ingest, analyze, prove, revise and friends (build plan Part 21).

runner.run_job      claim, heartbeat, run a handler, record success or failure
handlers            the body of each job kind
stages              stubs for the pipeline stages, wired in wave 2
spawn               starting a job on Modal (or a thread, or inline in tests)
lifecycle           failing jobs, and reaping jobs whose worker went quiet
worker.run_job_entry  what the Modal Functions in backend/modal_app.py call
"""
