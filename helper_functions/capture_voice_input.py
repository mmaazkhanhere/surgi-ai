import streamlit as st
import speech_recognition as sr

# Function to capture voice input and return transcribed text
# Function to capture voice input
def capture_voice_input():
    recognizer = sr.Recognizer()
    captured_text = ""

    with sr.Microphone() as source:
        st.write("Listening for your question (say 'please answer' to stop)...")
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                captured_text += " " + text
                st.write(f"You said: {text}")

                # If 'please answer' is detected, stop listening and process the question
                if "please answer" in text.lower():
                    st.write("End of question detected.")
                    break

            except sr.UnknownValueError:
                st.write("Sorry, I did not understand the audio.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")
                break

    return captured_text.replace("please answer", "").strip()  # Remove 'please answer' from the captured text