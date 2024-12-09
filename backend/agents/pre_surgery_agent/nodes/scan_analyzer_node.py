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

def scan_report_analyzer_node(state: PreSurgeryState):
    """LangGraph node that gives insights based on scans of the patient"""

    print('Scan Node')
    patient_history: str = state['patient_history']
    scan_reports: str = state.get('scan_report')
    operation: str = state.get('operation')

    instructions: str = """
        Analyze the provided medical scan reports to offer insights for a {operation} operation.  
        Consider the patient's history detailed below. Focus exclusively on the scan reports, providing 
        valuable insights that will be directly incorporated into a detailed surgical procedure report.

        **Patient History:**
        {patient_history}

        **Operation:**
        {operation}

        **Scan Reports:**
        {scan_reports}

        **Guidelines for Insights:**

        * **Specificity:**  Insights must be directly relevant to the operation and patient history.  
        Avoid general information.
        * **Actionability:** Insights should inform surgical planning and decision-making.
        * **Clarity:** Use precise medical terminology. Explain any complex findings clearly.
        * **Structure:**  Present insights as a numbered list with concise, bulleted points. Each point 
        should start with a strong verb (e.g., "Note," "Observe," "Consider").
        * **Objectivity:**  Base all insights on the provided scan data and patient history. Avoid 
        speculation or personal opinions.

        Output must be short and concise

        **Example Insight Format:**

        1. *Note:*  Calcification is observed near the gallbladder, potentially indicating prior 
        inflammation.
        2. *Consider:* The presence of adhesions from previous abdominal surgery may require careful 
        dissection during the procedure.
        3. *Observe:*  No significant abnormalities are detected in the surrounding liver parenchyma.

        **Insights:**
    """
    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
        scan_reports=scan_reports
    )

    response: BaseMessage = model.invoke(prompt)
    state['scan_report_analyzer_node'] = response.content
    return state
