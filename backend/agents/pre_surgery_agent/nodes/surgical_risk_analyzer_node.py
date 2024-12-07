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

def surgical_risk_analyzer_node(state: PreSurgeryState):
    """LangGraph node that gives insights on risks associated with surgery"""

    print('Risk Analyzer Node')
    patient_history: str = state['patient_history']
    operation: str = state['operation']

    instructions = """
        You are an expert in surgical risk analysis.  Your task is to provide a detailed analysis of the potential risks associated with a surgical procedure, considering the patient's history and the specific operation to be performed.
        Analyze the potential risks associated with the following operation, considering the provided patient history.  Your analysis should be detailed, and focus on potential complications, their likelihood given the patient's specific circumstances, and preventative measures.  Structure your output as follows:

        **1. Patient History Summary:** Briefly summarize the relevant aspects of the patient's history that increase or decrease the risk of complications.
        **2. Procedure-Specific Risks:**  List the potential risks associated with the operation itself, explaining the mechanisms of each risk.
        **3. Combined Risk Assessment:**  For each identified risk, analyze how the patient's history modifies the likelihood of that risk occurring (increased, decreased, or no significant change). Explain your reasoning.
        **4. Risk Mitigation Strategies:**  For each risk, recommend specific preventative measures or strategies to mitigate the risk, and justify these recommendations.
        **5. Overall Risk Profile:** Provide a concise summary of the overall risk profile for the patient undergoing this procedure, considering the combined effects of the patient's history and the inherent risks of the procedure.  Include a qualitative assessment of the overall risk (e.g., low, moderate, high).

        **Patient History:**
        {patient_history}

        **Operation to be Performed:**
        {operation}

        **Example Output:**

        **1. Patient History Summary:** The patient's history of diabetes increases the risk of wound infections and delayed healing.  Their history of hypertension necessitates careful monitoring of blood pressure during and after surgery.
        **2. Procedure-Specific Risks:**  The risk of bleeding, infection, and nerve damage are potential complications associated with the operation itself.
        **3. Combined Risk Assessment:** The patient’s diabetes increases the likelihood of infection. Hypertension could increase the risk of bleeding due to issues in blood pressure control.
        **4. Risk Mitigation Strategies:**  Prophylactic antibiotics are recommended for the patient due to diabetes to reduce the risk of infection. Close blood pressure monitoring should be done to manage hypertension.
        **5. Overall Risk Profile:** The overall risk profile for the patient is moderate.  The combination of the procedure's inherent risks and the patient's pre-existing conditions necessitates close monitoring and proactive risk mitigation strategies.

        **Your Analysis:**
    """
    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
    )

    response: BaseMessage = model.invoke(prompt)
    state["risk_analyzer_report"] = response.content
    return state
