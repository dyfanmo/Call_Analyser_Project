import os
import pickle
from models.disclaimer.model import disclaimer_verifier, get_text_similarity

speaker_segments_dir = os.path.join("test_data", "disclaimer")


def test_get_text_similairty_different():
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    assert round(get_text_similarity(text1, text2), 2) < 0.1


def test_get_text_similairty_same():
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quick brown fox jumps over the lazy dog."
    assert round(get_text_similarity(text1, text2), 2) == 1


def test_get_text_similairty_near_identical():
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A quick brown fox jumps over a lazy dog."
    assert round(get_text_similarity(text1, text2), 2) < 1


def test_disclaimer():
    with open(os.path.join(speaker_segments_dir, "with_disclaimer.pkl"), "rb") as f:
        speaker_segment_dict = pickle.load(f)

    assert disclaimer_verifier(speaker_segment_dict)


def test_no_disclaimer():
    with open(os.path.join(speaker_segments_dir, "no_disclaimer.pkl"), "rb") as f:
        speaker_segment_dict = pickle.load(f)

    assert not disclaimer_verifier(speaker_segment_dict)


def test_disclaimer_late_seconds():
    with open(os.path.join(speaker_segments_dir, "late_disclaimer_seconds.pkl"), "rb") as f:
        speaker_segment_dict = pickle.load(f)

    assert not disclaimer_verifier(speaker_segment_dict)


def test_disclaimer_late_segments():
    with open(os.path.join(speaker_segments_dir, "late_disclaimer_segments.pkl"), "rb") as f:
        speaker_segment_dict = pickle.load(f)

    assert not disclaimer_verifier(speaker_segment_dict)
