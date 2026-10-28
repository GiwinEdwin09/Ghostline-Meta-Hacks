"""Scene analysis with Muse Wonder (build plan Part 6) and Wonder's help for the SAM retry
ladder (Part 7)."""

from .analyze import (
    SceneInput,
    SceneNotMeasurable,
    SceneResult,
    analyze_scene,
    render_prompt,
    require_measurable,
    setup_tips,
)
from .rescue import BoxLocation, PromptSuggestion, locate_box, make_rescue_hooks, rephrase_prompt
from .schema import (
    Hypothesis,
    ObjectProperties,
    QualityIssue,
    SceneAnalysis,
    SheetInfo,
    TrackedObject,
    sam_prompt_problems,
)

__all__ = [
    "BoxLocation",
    "Hypothesis",
    "ObjectProperties",
    "PromptSuggestion",
    "QualityIssue",
    "SceneAnalysis",
    "SceneInput",
    "SceneNotMeasurable",
    "SceneResult",
    "SheetInfo",
    "TrackedObject",
    "analyze_scene",
    "locate_box",
    "make_rescue_hooks",
    "rephrase_prompt",
    "render_prompt",
    "require_measurable",
    "sam_prompt_problems",
    "setup_tips",
]
