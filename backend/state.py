from langgraph.graph.message import add_messages

from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

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

class Grade(BaseModel):
  grade: str = Field(description='Grade the relevance of the response')