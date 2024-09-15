import os
from preprocessing.utils import convert_audio_to_wav, convert_stereo_audio_to_mono, get_audio_duration


test_audio_dir = os.path.join("test_data", "audio")
wav_audio_path = os.path.join(test_audio_dir, "wav_audio.wav")
mp3_audio_path = os.path.join(test_audio_dir, "mp3_audio.mp3")


def test_get_audio_duration():
    expected_duration_seconds = 110
    assert int(get_audio_duration(wav_audio_path)) == expected_duration_seconds
