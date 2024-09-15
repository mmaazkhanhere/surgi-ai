# helper_functions/text_to_speech.py

import io
from gtts import gTTS

def text_to_speech(text):
    """
    Converts text to speech and returns a BytesIO object containing the audio.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected text to be a string, but got {type(text)}")
    tts = gTTS(text)
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)  # Reset the pointer to the beginning of the BytesIO object
    return audio_file