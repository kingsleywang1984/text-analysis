import pytest

from src.config import load_config


def test_config_requires_embedding_model(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL must be set"):
        load_config()


def test_config_reads_embedding_model_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    cfg = load_config()
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
