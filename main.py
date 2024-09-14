import streamlit as st

from crews.during_surgery_crew import during_surgery_crew

st.title('SurgiAI')

surgeony_query = st.text_input('Enter your query')
patient_history = st.text_area('Enter patient history')

btn = st.button('Ask')

if btn:
    response = during_surgery_crew(surgeony_query, patient_history)
    st.write(response)