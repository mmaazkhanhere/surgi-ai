from langgraph.graph.message import add_messages

from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

class Grade(BaseModel):
  grade: str = Field(description='Grade the relevance of the response')

class State(TypedDict):
  surgeon_query: str
  patient_history: str
  anatomy_response: str
  infection_prevention_response: str
  complication_response: str
  expert_surgeon_response: str
  insight_accumulator_response: str
  conversation: Annotated[list, add_messages]
  messages: Annotated[list, add_messages]
  relevance: str
  answer: str
  max_iteration: int
  current_iteration: int 

class PreSurgeryState(TypedDict):
  patient_history: str
  operation: str
  prescription_report: str
  scan_report: str
  lab_report: str
  prescription_report_analyzer_node: str
  lab_report_analyzer_node: str
  scan_report_analyzer_node: str
  instrumentation_report: str
  risk_analyzer_report: str
  anesthesia_consultant_report: str
  surgical_workflow_report: str
  emergency_protocol_advicer: str
  accumulator: str
