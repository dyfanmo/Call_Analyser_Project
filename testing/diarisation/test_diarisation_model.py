import os
import pickle
from models.diarisation.model import get_speaker_segments


diaraisation_data_dir = os.path.join("test_data", "diarisation")

with open(os.path.join(diaraisation_data_dir, "result_segments.pkl"), "rb") as f:
    result_segments = pickle.load(f)


speaker_segment_dict = get_speaker_segments(result_segments)


def test_get_speaker_segments_keys():
    assert "SPEAKER 1" in speaker_segment_dict and "SPEAKER 2" in speaker_segment_dict


def test_get_speaker_segments_speaker_first_segment_time():
    expected_start_time = 0
    expected_end_time = 25.04

    first_speaker_segment = speaker_segment_dict["SPEAKER 1"][0]
    assert first_speaker_segment["start"] == expected_start_time and first_speaker_segment["end"] == expected_end_time


def test_get_speaker_segments_speaker_last_segment_time():
    expected_start_time = 132.64
    expected_end_time = 133.8

    last_speaker_segment = speaker_segment_dict["SPEAKER 1"][-1]
    assert last_speaker_segment["start"] == expected_start_time and last_speaker_segment["end"] == expected_end_time
