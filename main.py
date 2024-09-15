import streamlit as st
import os

from crews.during_surgery_crew import during_surgery_crew
from crews.pre_surgery_crew import pre_surgery_crew
from helper_functions.ocr_helper import ocr_helper
from helper_functions.display_files_in_rows import display_files_in_rows
from helper_functions.convert_to_pdf import convert_to_pdf
from helper_functions.PDF_text_extractor import extract_text_from_pdf
from crews.pre_surgery_crew import pre_surgery_crew

from helper_functions.active_listening import active_listening


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
    patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
    patient_age = st.text_input("Patient age", placeholder="Enter patient age")


    # Patient history
    
    st.write("<h3></h3>", unsafe_allow_html=True)
    patient_history = st.text_area("Enter patient history", placeholder="Jan Doe is a ....")

    # Prescription file
    st.write("<h3 style='margin-top:28px'>Prescription</h3>", unsafe_allow_html=True)
    prescription_files = st.file_uploader("Upload Prescriptions (PDF, JPG, JPEG)", 
                                        type=["pdf", "jpg", "jpeg"], 
                                        accept_multiple_files=True)
    display_files_in_rows(prescription_files, "Uploaded Prescriptions")
    

    # Lab Reports
    st.write("<h3 style='margin-top:28px'>Lab Reports</h3>", unsafe_allow_html=True)
    lab_report_files = st.file_uploader("Upload Test Reports (PDF)", 
                                        type=["pdf"], 
                                        accept_multiple_files=True)
    display_files_in_rows(lab_report_files, "Uploaded Lab Reports")

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
        has_uploaded_files = len(prescription_files) > 0 and len(lab_report_files) > 0 and len(scan_files) > 0
        
        if not is_surgery_name_valid:
            st.error("Surgery Name is required.")
        elif not has_uploaded_files:
            st.error("At least one file must be uploaded in each: prescriptions, lab reports, and scans reports.")
        else:
            IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
            prescription_text = ""
            lab_report_text = ""
            scan_text = ""

            if prescription_files:
                prescription_text = ""
                for uploaded_file in prescription_files:
                    filename = uploaded_file.name
                    _, ext = os.path.splitext(filename)
                    ext = ext.lower()

                    if ext == '.pdf':
                        try:
                            text = extract_text_from_pdf(uploaded_file)
                            prescription_text += text + "\n\n\n\n"
                        except Exception as e:
                            st.error(f"Failed to extract PDF: {filename}. Error: {e}")
                    elif ext in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}:
                        try:
                            text = ocr_helper(uploaded_file, preprocess=True, use_gpu=False)
                            prescription_text += text + "\n\n\n\n"
                            st.success(f"Performed OCR on image: {filename}")
                        except Exception as e:
                            st.error(f"Failed to perform OCR on image: {filename}. Error: {e}")
                    else:
                        st.warning(f"Unsupported file type: {filename}")


            for file in lab_report_files:
                lab_report_text += extract_text_from_pdf(file) + "\n\n\n\n"

            for file in scan_files:
                scan_text += extract_text_from_pdf(file) + "\n\n\n\n"

            
            pre_surgery_report= pre_surgery_crew(surgery_name, patient_age , prescription_text, lab_report_text,scan_text)

            # Todo: Get response from AI and assign to response variable
            st.success("Report generated successfully!")
            

            # Assuming CrewOutput is a class type

            output_type = pre_surgery_report
            type_as_str = str(output_type)
            
            report_pdf_conversion = convert_to_pdf(type_as_str)

            st.download_button(
                label="Download Report",
                data=report_pdf_conversion,
                file_name="pre_surgery_report.pdf",
                mime="application/pdf"
            )
elif st.session_state.active_section == "During Surgery Voice Chat":
    st.header("During Surgery Voice Chat")
    st.write("<h6>Our crew of AI Agents will guide you during the surgery and answer your queries</h6>", unsafe_allow_html=True)

    st.write("<h3></h3>", unsafe_allow_html=True)
    patient_history_file = st.file_uploader('Upload Pre-Surgery report',
                            type=["pdf"], 
                            accept_multiple_files=False)


    if patient_history_file:

        # Convert the report to text
        patient_history = extract_text_from_pdf(patient_history_file)
        
        # Start the main loop

        active_listening(patient_history)

