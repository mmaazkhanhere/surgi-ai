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

def lab_report_analyzer_node(state: PreSurgeryState):
    """LangGraph node that gives insights based on lab reports of the patient"""

    print('Lab Report Analyzer Node')
    patient_history: str = state['patient_history']
    lab_report: str = state.get('lab_report')
    operation: str = state.get('operation')

    instructions = """
        Analyze the provided lab reports in the context of the patient's history and the upcoming operation.  
        Provide valuable insights that could inform surgical decisions and contribute to a detailed surgical 
        procedure report.

        **Input:**

        * **Lab Reports:** {lab_report}
        * **Patient History:** {patient_history}
        * **Operation:** {operation}

        **Guidelines for Analysis and Insights:**

        1. **Focus on Relevant Metrics:** Identify key lab values that are directly relevant to the planned operation. 
        For example, in a liver resection, focus on liver function tests (LFTs), coagulation profiles, and other relevant parameters.
        2. **Abnormal Values and Potential Implications:** Highlight any abnormal lab results and discuss their potential 
        implications for the operation.  For example, elevated creatinine levels might indicate reduced kidney function, 
        necessitating adjustments to anesthetic or fluid management.  Consider discussing the significance of these 
        findings in the context of the specific operation.
        3. **Patient-Specific Considerations:** Consider the patient's history and comorbidities when interpreting the 
        lab reports.  For example, a patient with diabetes might have different considerations regarding blood glucose 
        levels compared to a patient without diabetes.  How do the lab values inform the choices of anesthetic techniques 
        or postoperative care?
        4. **Influence on Surgical Planning:** Explain how your analysis of the lab values might inform the surgical approach.  
        For example, an elevated INR would influence the approach to the choice of medications, timing of the procedure or 
        clotting factors that may need to be administered.
        5. **Risk Assessment:** Assess the potential surgical risks based on the lab results.  For example, low hemoglobin 
        levels may increase the risk of blood loss.
        6. **Postoperative Considerations:** Note any specific considerations needed for postoperative management related to 
        lab results.  For example, certain lab values might suggest higher risks of bleeding and influence how often and how 
        the patient will be monitored.


        **Output Format:**

        Present your analysis in a clear, concise, and well-structured format.  Use bullet points or numbered lists for better 
        readability.  Each point should include:
        * The specific lab value(s) being discussed.
        * The numerical results (with units).
        * Your interpretation of the results, relating them to the planned operation and patient history.
        * The potential influence on surgical planning, risk assessment, or postoperative considerations.  For example:

        * **Insight 1:** Elevated creatinine levels (2.1 mg/dL) suggest reduced renal function. The patient’s history of 
        hypertension compounds this concern.  Careful hydration and monitoring of renal function post-operation will be necessary.
        * **Insight 2:**  Normal coagulation profile (INR 1.0) indicates no contraindication for the planned surgery.  
        Routine pre-operative evaluation for blood loss mitigation is advised.

    """
    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
        lab_report=lab_report
    )

    response: BaseMessage = model.invoke(prompt)
    state['lab_report_analyzer_node'] = response.content
    return state
