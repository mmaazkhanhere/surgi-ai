import streamlit as st

from crews.post_surgery_faqs_crew import surgery_post_faq_crew

st.title('Post Surgery Report Creator')

# patient_info = st.text_area('Patient Information')
surgery_details = st.text_area('Surgery Details')
surgeon_conversation = st.text_area('Surgeon Conversation')
btn = st.button('Generate Report')

if btn:
    response = surgery_post_faq_crew(surgery_details, surgeon_conversation)
    st.write(response)