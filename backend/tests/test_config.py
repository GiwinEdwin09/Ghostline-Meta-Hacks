import pytest

from ghostline.config import get_settings


def test_cors_origins_are_split_and_trimmed(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example ,")

    assert get_settings().cors_origins == ["https://a.example", "https://b.example"]


def test_unknown_llm_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "sometimes")

    with pytest.raises(ValueError):
        get_settings()


def test_env_example_names_every_setting_the_code_reads():
    """A setting that isn't in .env.example can't be found by whoever deploys this."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = (root / "backend/ghostline/config.py").read_text()
    read = set(re.findall(r'env\("([A-Z_0-9]+)"', config))
    read |= {"LLM_CACHE_DIR", "GHOSTLINE_KEEP_WARM"}  # read outside config.py
    example = (root / ".env.example").read_text()
    listed = set(re.findall(r"^#?\s*([A-Z][A-Z_0-9]+)=", example, re.M))

    assert read - listed == set()
    for line in example.splitlines():
        name, _, value = line.partition("=")
        if name in {"MODEL_API_KEY", "REVIEWER_API_KEY", "DATABASE_URL", "APP_PASSCODE"}:
            assert value == "", f"{name} must not hold a value in .env.example"
