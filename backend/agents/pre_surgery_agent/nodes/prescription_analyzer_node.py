import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from state import PreSurgeryState

load_dotenv()

model = ChatGroq(
    model="llama-3.2-1b-preview",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

def prescription__report_analyzer_node(state: PreSurgeryState):
    """LangGraph node that gives insights based on scans of the patient"""

    print('Prescription Node')
    patient_history: str = state['patient_history']
    prescription_report: str = state.get('prescription_report')
    operation: str = state.get('operation')

    instructions = """Analyze the provided `prescription_report` to extract valuable insights relevant to the specified operation, considering the  
        patient's history.  Focus exclusively on the prescription report and disregard any other information provided.  

        **Guidelines for the Output:**

        1. **Direct Relevance:**  Only provide insights directly related to the medication(s) listed, their potential impact on the planned surgery, or any contraindications.
        2. **Clarity and Conciseness:** Present the insights in a clear, concise manner, using bullet points for each key finding.
        3. **Actionable Information:** Frame the insights to be actionable for a surgeon writing a surgical procedure report.  Mention specific concerns or precautions the surgeon should take during or after the procedure due to the medications identified.
        4. **Specific Operation Focus:**  The insights should explicitly relate to the operation to be performed.
        5. **Avoid Speculation:** Do not make assumptions about the patient's current health status beyond what is directly stated in the prescription report.
        6. **Structure:** Follow this structure for each insight:
        * **Medication:** Name of the medication, dosage, and route of administration (if available).
        * **Potential Impact/Concern:**  Explain how the medication might influence the surgery, recovery, or potential complications.
        * **Actionable Recommendation:** Provide specific recommendations for the surgeon.


        **Inputs:**

        * `prescription_report`:  {prescription_report}
        * `patient_history`: {patient_history}
        * `operation`: {operation}

        **Example Output:**

        * **Medication:** Warfarin (5mg daily, oral)
        * **Potential Impact/Concern:** Increased risk of bleeding during surgery due to anticoagulant properties.
        * **Actionable Recommendation:** Consider holding warfarin for [number] days prior to the procedure. Consult with a hematologist to determine the appropriate management plan. Closely monitor coagulation parameters during and after surgery.

        **Output:**
    """
    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
        prescription_report=prescription_report
    )

    response: BaseMessage = model.invoke(prompt)
    state['lab_report_analyzer_node'] = response.content
    return state
