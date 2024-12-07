from langgraph.graph.state import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from .nodes.accumulator_node import accumulator_node
from .nodes.anesthesia_consultant_node import anesthesia_consultant_node
from .nodes.emergency_protocol_advisor_node import emergency_protocol_advisor_node
from .nodes.instrumentaion_decider_node import instrumentation_decider_node
from .nodes.lab_report_analyzer_node import lab_report_analyzer_node
from .nodes.prescription_analyzer_node import prescription__report_analyzer_node
from .nodes.surgical_risk_analyzer_node import surgical_risk_analyzer_node
from .nodes.surgical_workflow_node import surgical_workflow_node
from .nodes.scan_analyzer_node import scan_report_analyzer_node

from state import PreSurgeryState

def pre_surgical_report_agent(state: PreSurgeryState):
    builder = StateGraph(PreSurgeryState)

    builder.add_node("Prescription Analyzer", prescription__report_analyzer_node)
    builder.add_node("Scan Analyzer", scan_report_analyzer_node)
    builder.add_node("Lab Report Analyzer", lab_report_analyzer_node)
    builder.add_node("Instrument Decider", instrumentation_decider_node)
    builder.add_node("Risk Analyzer", surgical_risk_analyzer_node)
    builder.add_node("Anesthesia Consultant", anesthesia_consultant_node)
    builder.add_node("Emergency Protocol Advisor", emergency_protocol_advisor_node)
    builder.add_node("Accumulator", accumulator_node)
    builder.add_node("Surgical Workflow", surgical_workflow_node)

    builder.add_edge(START, "Prescription Analyzer")
    builder.add_edge("Prescription Analyzer", "Scan Analyzer")
    builder.add_edge("Scan Analyzer", "Lab Report Analyzer")
    builder.add_edge("Lab Report Analyzer", "Instrument Decider")
    builder.add_edge("Instrument Decider", "Risk Analyzer")
    builder.add_edge("Risk Analyzer", "Anesthesia Consultant")
    builder.add_edge("Anesthesia Consultant", "Emergency Protocol Advisor")
    builder.add_edge("Emergency Protocol Advisor", "Accumulator")
    builder.add_edge("Accumulator", "Surgical Workflow")

    agent = builder.compile()
    result = agent.invoke(state)
    return result["surgical_workflow_report"]