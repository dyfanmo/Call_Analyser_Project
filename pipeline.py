import os
import warnings
import openai
from argparse import ArgumentParser
from models.speech.model import transcribe_audio
from models.diarisation.model import get_speaker_segments
from models.disclaimer.model import disclaimer_verifier
from models.sentiment.model import return_call_sentiment
from models.summariser.gpt_model import get_gpt_call_summary
from preprocessing.utils import convert_audio_to_wav, is_audio_mono, convert_stereo_audio_to_mono


warnings.filterwarnings("ignore")


def main():
    parser = ArgumentParser()
    parser.add_argument("--audio_path")
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--openai_key", default="")
    args = parser.parse_args()

    cleaned_data_dir = "cleaned_data"
    os.makedirs(cleaned_data_dir, exist_ok=True)

    new_audio_path = convert_audio_to_wav(args.audio_path, cleaned_data_dir)

    if not is_audio_mono(new_audio_path):
        new_audio_path = convert_stereo_audio_to_mono(new_audio_path, cleaned_data_dir)

    audio_transcription = transcribe_audio(new_audio_path, args.num_speakers)
    speaker_segment_dict = get_speaker_segments(audio_transcription["segments"])

    disclaimer_verified = disclaimer_verifier(speaker_segment_dict)
    call_satisfaction = return_call_sentiment(audio_transcription["text"])

    print("Disclaimer Verified:", disclaimer_verified)
    print("Call Satifaction:", call_satisfaction)

    if args.openai_key:
        client = openai.OpenAI(api_key=args.openai_key)
        response = get_gpt_call_summary(client, str(speaker_segment_dict))
        print("Call Summary:\n")
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
