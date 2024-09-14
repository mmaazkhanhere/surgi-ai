import streamlit as st
from datetime import date

# Set the page configuration
st.set_page_config(page_title="SurgiAI", layout="centered")

# Title of the application
st.title("SurgiAI")

# Sidebar for Navigation using buttons
st.sidebar.title("Navigation")
if "active_section" not in st.session_state:
    st.session_state["active_section"] = "Pre Surgery Report"  # Default section

# Sidebar buttons for navigation
if st.sidebar.button("Pre Surgery Report"):
    st.session_state["active_section"] = "Pre Surgery Report"
if st.sidebar.button("During Surgery Voice Chat"):
    st.session_state["active_section"] = "During Surgery Voice Chat"
if st.sidebar.button("Post Surgery Suggestions"):
    st.session_state["active_section"] = "Post Surgery Suggestions"
if st.sidebar.button("About"):
    st.session_state["active_section"] = "About"

# Helper function to display files in rows with 3 files per row
def display_files_in_rows(file_list, header):
    if file_list:
        st.subheader(header)
        for i in range(0, len(file_list), 3):
            cols = st.columns(3)  # Create 3 columns
            for j, file in enumerate(file_list[i:i+3]):  # Display up to 3 files per row
                with cols[j]:
                    st.write(file.name)

# State variable to track if the report is generated
if "report_generated" not in st.session_state:
    st.session_state["report_generated"] = False

# Handling each section logic based on the active section
active_section = st.session_state["active_section"]

if active_section == "Pre Surgery Report":
    st.header("Pre Surgery Report")

    # Input Surgery Name
    surgery_name = st.text_input("Surgery Name", placeholder="Enter surgery name")

    # File upload sections
    st.header("Prescriptions")
    prescription_files = st.file_uploader("Upload Prescriptions (PDF, DOCX, DOC, or Images)", 
                                          type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], 
                                          accept_multiple_files=True)
    display_files_in_rows(prescription_files, "Uploaded Prescriptions")

    st.header("Lab Reports")
    test_report_files = st.file_uploader("Upload Test Reports (PDF, DOCX, DOC, or Images)", 
                                         type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], 
                                         accept_multiple_files=True)
    display_files_in_rows(test_report_files, "Uploaded Lab Reports")

    st.header("Scans")
    scan_files = st.file_uploader("Upload Scans (PDF, DOCX, DOC, or Images)", 
                                  type=["pdf", "docx", "doc", "png", "jpg", "jpeg"], 
                                  accept_multiple_files=True)
    display_files_in_rows(scan_files, "Uploaded Scans")

    # Generate Report Button
    if st.button("Generate Report"):
        # Validation checks
        is_surgery_name_valid = surgery_name.strip() != ""
        has_uploaded_files = len(prescription_files) > 0 and len(test_report_files) > 0 and len(scan_files) > 0
        
        if not is_surgery_name_valid:
            st.error("Surgery Name is required.")
        elif not has_uploaded_files:
            st.error("At least one file must be uploaded in each: prescriptions, lab reports, and scans.")
        else:
            st.session_state["report_generated"] = True
            st.success("Report generated successfully!")
            # Add backend integration logic here

elif active_section == "During Surgery Voice Chat":
    st.header("During Surgery Voice Chat")
    st.write("Voice chat functionality can be integrated here.")

elif active_section == "Post Surgery Suggestions":
    st.header("Post Surgery Suggestions")
    st.write("Post surgery suggestions will go here.")

elif active_section == "About":
    st.header("About SurgiAI")
    st.write("""
        SurgiAI is a revolutionary AI-powered application designed to assist surgeons with pre-surgery reporting, 
        intra-surgery communication, and post-surgery suggestions. It provides a seamless and efficient solution to 
        streamline surgical procedures and documentation.
    """)
    st.write("Version: 1.0")
    st.write("Developed by GenAgents.")