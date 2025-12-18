from __future__ import annotations

import re
import unicodedata

_RE_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Deterministic text normalization:
    - Unicode normalize (NFKC)
    - Trim
    - Collapse repeated whitespace to single spaces
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.strip()
    t = _RE_WS.sub(" ", t)
    return t


