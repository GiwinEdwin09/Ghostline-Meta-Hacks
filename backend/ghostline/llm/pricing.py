"""Prices in USD (build plan section 0, Part 23) and the cost formulas built on them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    """USD per 1M tokens."""

    input: float
    cached: float
    output: float

    def cost(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        """Cost of one call. Output tokens include reasoning tokens; cached tokens are a
        subset of the input tokens."""
        fresh = max(input_tokens - cached_tokens, 0)
        return (
            fresh * self.input + cached_tokens * self.cached + output_tokens * self.output
        ) / 1e6


WONDER = TokenPrice(input=4.00, cached=0.20, output=20.00)
SPARK = TokenPrice(input=1.25, cached=0.15, output=4.25)

SAM_VIDEO_USD_PER_FRAME = 0.20 / 1000
SAM_VIDEO_MIN_FRAMES = 50
SAM_IMAGE_USD = 2.50 / 1000


def price_for_model(model: str) -> TokenPrice:
    """Spark models price as Spark; anything else prices as Wonder, the dearer rate."""
    return SPARK if model.startswith("muse-spark") else WONDER


def sam_video_cost(frames: int) -> float:
    """Video is billed per frame with a 50-frame minimum per request."""
    return max(frames, SAM_VIDEO_MIN_FRAMES) * SAM_VIDEO_USD_PER_FRAME


def sam_image_cost() -> float:
    return SAM_IMAGE_USD
