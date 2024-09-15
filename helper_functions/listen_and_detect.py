import streamlit as st
from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Function to detect voice input and handle session closing based on "that's it"
def listen_and_detect():
    vosk_model = Model(r"vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(vosk_model, 16000)
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    st.write("Listening for either 'I have a question' or 'that's it' to close the session...")

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if "i have a question" in text.lower():
                st.write("Wake word detected! Ready to capture your question.")
                return "question"
            if "that's it" in text.lower():
                st.write("Session closure detected. Ending session...")
                st.stop()  # End the Streamlit session
                return "exit"
        else:
            continue