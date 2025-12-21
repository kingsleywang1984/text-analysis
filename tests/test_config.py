import pytest

from src.config import load_config


def test_config_requires_embedding_model(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL must be set"):
        load_config()
