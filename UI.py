import streamlit as st

from crews.post_surgery_checklist_crew import post_surgery_checklist_crew

st.title('Post Surgery Report Creator')

patient_condition = st.text_area('Patient Condition')
surgery_details = st.text_area('Surgery Details')
surgeon_conversation = st.text_area('Surgeon Conversation')
btn = st.button('Generate Report')

if btn:
    response = post_surgery_checklist_crew(surgery_details, surgeon_conversation, patient_condition)
    st.write(response)