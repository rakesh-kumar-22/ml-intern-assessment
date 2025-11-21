import pytest
from src.ngram_model import TrigramModel


def test_fit_and_generate():
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated_text = model.generate()

    # Should return a non-empty string
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 0


def test_empty_text():
    model = TrigramModel()
    text = ""
    model.fit(text)

    generated_text = model.generate()

    # For empty text, model has no trigrams → generate should return ""
    assert generated_text == ""


def test_short_text():
    model = TrigramModel()
    text = "I am."
    model.fit(text)

    generated_text = model.generate()

    # Should return a string (may be empty or 1–2 tokens)
    assert isinstance(generated_text, str)
    