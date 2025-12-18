from src.pipeline.normalize import normalize_text


def test_normalize_text_trims_and_collapses_whitespace():
    assert normalize_text("  hello   world \n") == "hello world"


def test_normalize_text_nfkc_and_empty():
    # full-width forms -> ASCII under NFKC
    assert normalize_text("ＡＢＣ") == "ABC"
    assert normalize_text("   \n\t  ") == ""


