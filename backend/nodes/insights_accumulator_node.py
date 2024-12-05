import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from state import State

load_dotenv()

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    verbose=True,
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

def insight_accumulator_node(state: State):
  """LangGraph node that accumulates all the insights from specialized agents and
  give a final output related to surgeon query"""

  print("Insight Accumulator")
  surgeon_query: str = state['surgeon_query']
  patient_history: str = state['patient_history']
  anatomy_response: str = state['anatomy_response']
  infection_prevention_response: str = state['infection_prevention_response']
  complication_response: str = state['complication_response']
  surgeon_response: str = state['expert_surgeon_response']

  instructions: str = """
    You are an expert surgical insight accumulator. Your task is to gather insights from the
    provided information and accumulate them into actionable insights that can be used
    to respond to surgeon query {surgeon_query}.

    **Context:**

    * Patient History: {patient_history}
    * Relevant Anatomy: {anatomy_response}
    * Infection Prevention Measures: {infection_prevention_response}
    * Potential Complications and their Management: {complication_response}
    * Expert Surgeon Response: {surgeon_response}

    **Task:**

    Carefully analyze the provided information, considering the patient's specific
    history and the potential surgical challenges. Identify key insights that may
    influence the surgical approach, including but not limited to:

    * **Critical anatomical variations:**  Are there any unusual anatomical structures
    or relationships that require special attention?
    * **High-risk infection areas:** Are there any specific areas where infection
    prevention measures need to be reinforced?
    * **Anticipated complications and mitigation strategies:** What are the most
    likely complications, and how can they be effectively prevented or managed?
    * **Alternative surgical techniques:** Based on the available information, are
    there alternative surgical techniques that could be considered?

    Synthesize these insights into a concise and actionable summary for the next surgical planning node.
    The summary should clearly highlight the key considerations derived from the provided data and
    the patient's specific situation.  Ensure the insights are specific and directly relevant to
    guiding the surgical plan.

    **Output Format:**

    Provide a structured summary of insights, using bullet points or a numbered list.
    Avoid redundancy and focus on the most critical insights.

    **Example:**

    * **Insight 1:** Due to patient's history of [specific condition], the surgical
    approach should prioritize [specific technique] to minimize risk of [specific complication].
    * **Insight 2:** The presence of [specific anatomical variation] necessitates
    careful dissection near the [anatomical region] to avoid [potential complication].

    **Insights:**
  """
  prompt: str = instructions.format(
    surgeon_query=surgeon_query,
    patient_history=patient_history,
    anatomy_response=anatomy_response,
    infection_prevention_response=infection_prevention_response,
    complication_response=complication_response,
    surgeon_response=surgeon_response,
  )

  response: BaseMessage = model.invoke(prompt)
  return {"insight_accumulator_response": response.content}

