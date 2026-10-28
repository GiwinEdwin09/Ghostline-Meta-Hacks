import pytest

from ghostline.config import get_settings


def test_cors_origins_are_split_and_trimmed(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example ,")

    assert get_settings().cors_origins == ["https://a.example", "https://b.example"]


def test_unknown_llm_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "sometimes")

    with pytest.raises(ValueError):
        get_settings()
