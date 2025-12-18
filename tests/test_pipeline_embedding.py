import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.models.schemas import AnalyzeRequest
from src.pipeline.embedding import TfidfEmbeddingProvider

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(filename: str) -> dict:
    p = DATA_DIR / filename
    assert p.exists(), f"Missing example file: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def test_embedding_provider_shapes_and_finite_values_from_task_example():
    payload = _load_json("input_example.json")
    req = AnalyzeRequest.model_validate(payload)

    texts = [x.sentence for x in req.baseline[:5]]  # keep test fast
    provider = TfidfEmbeddingProvider(load_config())

    emb = provider.embed(texts)

    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 2
    assert emb.shape[0] == len(texts)
    assert emb.shape[1] > 0
    assert np.isfinite(emb).all()


def test_embedding_is_normalized():
    payload = _load_json("input_example.json")
    req = AnalyzeRequest.model_validate(payload)

    texts = [x.sentence for x in req.baseline[:3]]
    provider = TfidfEmbeddingProvider(load_config())
    emb = provider.embed(texts)

    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
