import os
import wave
import contextlib
import subprocess
from pydub import AudioSegment


def convert_audio_to_wav(path, output_dir):
    if path[-3:] != "wav":
        new_path = os.path.join(output_dir, "wav_data.wav")
        try:
            subprocess.call(["ffmpeg", "-i", path, new_path, "-y"])
        except:
            return path, "Error: Could not convert file to .wav"
        path = new_path
    return path


def convert_stereo_audio_to_mono(path, output_dir):
    new_audio_path = os.path.join(output_dir, f"mono_audio.wav")
    sound = AudioSegment.from_wav(path)
    sound = sound.set_channels(1)
    sound.export(new_audio_path, format="wav")
    return new_audio_path


def is_audio_mono(path):
    return True if AudioSegment.from_wav(path).channels == 1 else False


def get_audio_duration(path):
    with contextlib.closing(wave.open(path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def load_text_file(path):
    with open(path, "r") as file:
        return file.read().replace("\n", "")
