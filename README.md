# Call Analyser Project

## Requirements
Use an environment with Python 3.12, then

`pip install -r requirements.txt`

### Installing ffmpeg

Before running the pipeline, you'll need to install ffmpeg for the audio proeccessing 
#### For Windows:

1. Visit the [ffmpeg website](https://ffmpeg.org/download.html) and download the latest version for Windows.
2. Extract the downloaded ZIP file to a location on your computer.
3. Add the bin directory inside the extracted folder to your system's PATH environment variable. 

#### For macOS:

   ```bash
   brew install ffmpeg
```

#### For Linux:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```


## Running the pipeline
The `pipeline.py` script will convert a raw audio conversation into sentiment analysis class, verify if an agent read a disclaimer and a summarise the conversation.

`python pipeline.py --audio_path example.mp3 --num_speakers <int> --openai_key <YOUR API KEY> `

We've included an example image at `Call-1-Example.mp3`, but you can replace this with any audio.

## Repo structure
Models used in the pipeline are contained in `models/`

I've included tests for the main models and the preprocessing steps in `testing/`. The `pipeline.py` script serves an integration test for the whole pipeline.
