from models.disclaimer.constants import (
    similarity_thresehold,
    disclaimer_text,
    max_start_time_seconds,
    max_speaker_segments,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("paraphrase-distilroberta-base-v1")


def get_text_similarity(text1, text2):
    sentence_embeddings = model.encode([text1, text2])
    return cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[0][0]


def disclaimer_verifier(speaker_segments_dict):
    similarity_scores = []
    for speaker, segments in speaker_segments_dict.items():
        for i, segment in enumerate(segments):
            if segment["start"] < max_start_time_seconds and i + 1 <= max_speaker_segments:
                similarity_scores.append(get_text_similarity(segment["text"], disclaimer_text))

    final_score = max(similarity_scores, default=0)
    return final_score > similarity_thresehold
