"""The Ghostline benchmark (build plan Part 25): ground truth, the audio method for bounces, and
the runner that scores the deployed pipeline against them.

    python -m bench.run --replay          # against a back end in LLM_MODE=replay (free)
    python -m bench.run --live --yes      # against a live back end (about $1.50 per clip)
    python -m bench.audio_bounce drop1.wav drop2.wav ...
"""
