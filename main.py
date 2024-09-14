import streamlit as st
from crews.during_surgery_crew import during_surgery_crew
from helper_functions.display_files_in_rows import display_files_in_rows
from helper_functions.convert_to_pdf import convert_to_pdf

st.set_page_config(
    page_title="SurgiAI",
    page_icon="stethoscope",
)

# Sidebar for Navigation using buttons
st.sidebar.title("SurgiAI")
if "active_section" not in st.session_state:
    st.session_state.active_section = "Pre Surgery Report"  # Default section

# Sidebar buttons for navigation
if st.sidebar.button("Pre Surgery Report"):
    st.session_state["active_section"] = "Pre Surgery Report"
if st.sidebar.button("During Surgery Voice Chat"):
    st.session_state["active_section"] = "During Surgery Voice Chat"
if st.sidebar.button("Post Surgery Suggestions"):
    st.session_state["active_section"] = "Post Surgery Suggestions"
if st.sidebar.button("About"):
    st.session_state["active_section"] = "About"


if st.session_state.active_section == "Pre Surgery Report":
    st.header("Pre Surgery Report")
    st.write("<h6>This section is for generating a pre-surgery report for the surgery. Upload the relevant documents to get complete report</h6>", unsafe_allow_html=True)

    st.write("<h3></h3>", unsafe_allow_html=True)
    
    # Input Surgery Name
    surgery_name = st.text_input("Surgery Name", placeholder="Enter surgery name")

    # Prescription file
    st.write("<h3 style='margin-top:28px'>Prescription</h3>", unsafe_allow_html=True)
    prescription_files = st.file_uploader("Upload Prescriptions (PDF)", 
                                        type=["pdf"], 
                                        accept_multiple_files=True)
    display_files_in_rows(prescription_files, "Uploaded Prescriptions")

    # Lab Reports
    st.write("<h3 style='margin-top:28px'>Lab Reports</h3>", unsafe_allow_html=True)
    test_report_files = st.file_uploader("Upload Test Reports (PDF)", 
                                        type=["pdf"], 
                                        accept_multiple_files=True)
    display_files_in_rows(test_report_files, "Uploaded Lab Reports")

    # Scan Reports
    st.write("<h3 style='margin-top:28px'>Scans Report</h3>", unsafe_allow_html=True)
    scan_files = st.file_uploader("Upload Scans (PDF)", 
                                type=["pdf"], 
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
            st.error("At least one file must be uploaded in each: prescriptions, lab reports, and scans reports.")
        else:
            # Todo: Get response from AI and assign to response variable
            response = "This is a sample response from the AI model."
            st.success("Report generated successfully!")
            st.write(response)

            report_pdf_conversion = convert_to_pdf(response)

            st.download_button(
                label="Download Report",
                data=report_pdf_conversion,
                file_name="pre_surgery_report.pdf",
                mime="application/pdf"
            )
            
