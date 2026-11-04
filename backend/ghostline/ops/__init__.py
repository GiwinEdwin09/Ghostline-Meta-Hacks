"""Operations (build plan Parts 27 and 28): the nightly backup, the 14-day retention purge, and
the prompt version that stamps benchmark and eval results.

The functions here take a Repo, a VolumePaths and a VolumeSync, like store/retention.py, so they
run the same from a Modal scheduled function (ghostline.ops.entry), a laptop (python -m
ghostline.ops) and a test. docs/runbook.md says when to run each by hand.
"""

from ghostline.ops.versions import PromptVersion, prompt_version

__all__ = ["PromptVersion", "prompt_version"]
