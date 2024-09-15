import streamlit as st
import base64

# Function to play audio without autoplay using HTML
def play_audio(audio_data):
    audio_data.seek(0)  # Ensure we are at the beginning of the BytesIO buffer
    audio_bytes = audio_data.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)