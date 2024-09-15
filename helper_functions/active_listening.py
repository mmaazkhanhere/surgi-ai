import streamlit as st

from crews.during_surgery_crew import during_surgery_crew
from langchain.schema import HumanMessage, AIMessage

from helper_functions.listen_and_detect import listen_and_detect
from helper_functions.play_audio import play_audio
from helper_functions.text_to_speech import text_to_speech
from helper_functions.capture_voice_input import capture_voice_input


def active_listening(patient_history):
            if "messages" not in st.session_state:
                st.session_state.messages = []

        #Display previous chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Voice Input Section placed at the bottom
            st.write("---")
            while True:

                detected_action = listen_and_detect()
                if detected_action == "question":
                    # Capture user's question after wake word is detected
                    voice_input_text = capture_voice_input()
            
                    # Display user's message and append to session state
                    if voice_input_text:
                        human_message = HumanMessage(content=voice_input_text)
                        st.session_state.messages.append(human_message)

                        with st.chat_message("user"):
                            st.markdown(human_message.content)

                        # Pass the content directly if the function expects a string
                        ai_response_content = during_surgery_crew(voice_input_text, patient_history)  

                        ai_response_content = str(ai_response_content)    
                        # Append AI response to session state
                        ai_message = AIMessage(content=ai_response_content)
                        st.session_state.messages.append(ai_message)

                        # Display AI response
                        with st.chat_message("assistant"):
                            st.markdown(ai_message.content)

                        # Convert AI response to speech and play it automatically
                        response_audio = text_to_speech(ai_message.content)
                        audio_bytes = response_audio.read()
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)