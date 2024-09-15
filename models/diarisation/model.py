import torch
import numpy as np
from pyannote.core import Segment
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
from preprocessing.utils import get_audio_duration

audio = Audio()
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cpu"))


def segment_embedding(path, segment, duration):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])


def make_embeddings(path, segments, duration):
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(path, segment, duration)
    return np.nan_to_num(embeddings)


def add_speaker_labels(segments, embeddings, num_speakers):
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i, segment in enumerate(segments):
        segment["speaker"] = f"SPEAKER {labels[i] + 1}"


def add_speaker_labels_to_segments(path, segments, num_speakers):
    duration = get_audio_duration(path)
    num_speakers = min(max(round(num_speakers), 1), len(segments))
    if len(segments) == 1:
        segments[0]["speaker"] = "SPEAKER 1"
    else:
        embeddings = make_embeddings(path, segments, duration)
        add_speaker_labels(segments, embeddings, num_speakers)


def append_speaker_segment(speaker_dict, segment, text, segment_length, speaker_start_time):
    speaker_segments = speaker_dict.setdefault(segment["speaker"], [])
    speaker_segments.append(
        {
            "text": text.strip(),
            "start": speaker_start_time,
            "end": segment["end"],
            "segment_num": segment_length,
        }
    )


def get_speaker_segments(segments):
    text = ""
    segment_length = 0
    speaker_dict = {}

    for i in range(1, len(segments)):
        current_segment = segments[i]
        previous_segment = segments[i - 1]

        segment_length += 1
        text += previous_segment["text"] + " "

        if current_segment["speaker"] != previous_segment["speaker"]:
            start_time = segments[i - segment_length]["start"]
            append_speaker_segment(speaker_dict, previous_segment, text, segment_length, start_time)

            text = ""
            segment_length = 0

    text += current_segment["text"] + " "
    start_time = segments[i - segment_length]["start"]
    append_speaker_segment(speaker_dict, current_segment, text, segment_length, start_time)
    return speaker_dict
