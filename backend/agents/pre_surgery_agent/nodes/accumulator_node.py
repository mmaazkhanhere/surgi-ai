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

def accumulator_node(state: PreSurgeryState):
    """LangGraph node that gives insights related to potential emergency in surgery"""

    print('Accumulator Node')
    patient_history: str = state.get('patient_history')
    operation: str = state.get('operation')
    scan_report_analyzer_node: str = state.get('scan_report_analyzer_node')
    lab_report_analyzer_node: str = state.get('lab_report_analyzer_node')
    prescription_report_analyzer_node: str = state.get('prescription_report_analyzer_node')
    instrumentation_report: str = state.get('instrumentation_report')
    risk_analyzer_report: str = state.get('risk_analyzer_report')
    anesthesia_consultant_report: str = state.get('anesthesia_consultant_report')
    emergency_protocol_advicer: str = state.get('emergency_protocol_advicer')

    print(f'Emergency Protocol: {emergency_protocol_advicer[:50]}')

    instructions = """
        You are an AI tasked with generating a comprehensive and detailed surgical procedure report based on the inputs from multiple specialized nodes. Accumulate and synthesize the following information:

        Scan Analyzer {scan_report_analyzer_node}: Insights from imaging reports, including anatomical landmarks, abnormalities, and areas requiring intervention.
        Lab Report Analyzer {lab_report_analyzer_node}: Key findings from lab results, highlighting any conditions or abnormalities relevant to the procedure.
        Prescription Analyzer {prescription_report_analyzer_node}: Details about current medications and their potential impact on the surgery or anesthesia plan.
        Instrumentation Decider {instrumentation_report}: Recommended surgical tools and equipment required for the procedure, considering patient-specific needs.
        Risk Analyzer {risk_analyzer_report}: Identified risks, their probabilities, and strategies to mitigate these risks during the surgery.
        Anesthesia Consultant {anesthesia_consultant_report}: Suggested anesthesia plan, including type, dosage, and monitoring requirements.
        Emergency Protocol Advisor {emergency_protocol_advicer}: Contingency plans for potential intraoperative complications or emergencies.
        Combine all this data into a single, cohesive output that details:

        Step-by-step surgical guidance for the surgery {operation}, considering the patient history {patient_history}.
        Necessary preparations and precautions.
        Recommended tools and techniques.
        Risk management strategies and emergency response protocols.
        The output should be precise, actionable, and tailored to ensure the surgery is conducted with minimal risk and high success probability."
    """

    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
        scan_report_analyzer_node=scan_report_analyzer_node,
        lab_report_analyzer_node=lab_report_analyzer_node,
        prescription_report_analyzer_node=prescription_report_analyzer_node,
        instrumentation_report=instrumentation_report,
        risk_analyzer_report=risk_analyzer_report,
        anesthesia_consultant_report=anesthesia_consultant_report,
        emergency_protocol_advicer=emergency_protocol_advicer
    )

    response: BaseMessage = model.invoke(prompt)
    state["accumulator"] = response.content
    return state
