import whisper
from models.diarisation.model import add_speaker_labels_to_segments


speech_model = whisper.load_model("base")


def transcribe_audio(audio_path, num_speakers):
    result = speech_model.transcribe(audio_path)
    segments = result["segments"]
    num_speakers = min(max(round(num_speakers), 1), len(segments))
    add_speaker_labels_to_segments(audio_path, segments, num_speakers)

    return result
