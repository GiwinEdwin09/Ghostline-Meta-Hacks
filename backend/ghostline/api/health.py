"""Liveness check for the API."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from ghostline import __version__
from ghostline.config import get_settings

router = APIRouter()


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    llm_mode: str


@router.get("/health")
def health() -> Health:
    # Database, Volume and Meta API checks join this once those parts exist (build plan Part 21).
    return Health(status="ok", version=__version__, llm_mode=get_settings().llm_mode)
