import streamlit as st

from crews.post_surgery_crew import operative_report_crew

st.title('Post Surgery Report Creator')

patient_info = st.text_area('Patient Information')
surgery_details = st.text_area('Surgery Details')
surgeon_conversation = st.text_area('Surgeon Conversation')
btn = st.button('Generate Report')

if btn:
    response = operative_report_crew(surgery_details, surgeon_conversation, patient_info)
    st.write(response)