import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from state import State

load_dotenv()

model = ChatGroq(
    model="llama-3.2-1b-preview",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

def infection_prevention_node(state: State):
  """LangGraph node that gives instruction to prevent infection"""

  print("Infection Prevention")
  surgeon_query: str = state['surgeon_query']
  patient_history: str = state['patient_history']
  instructions: str  = """
    You are a Certified Infection Preventionist (CIP) providing expert guidance to a surgeon.
    A surgical error has occurred, and the surgeon is seeking your immediate advice on
    infection prevention steps.  Your responses must be precise, evidence-based, and
    prioritize patient safety.

    **Context:**

    * **Patient History:** {patient_history}  (Provide a detailed patient history,
    including relevant medical conditions, allergies, past surgeries, and medications.
    Include any known immunodeficiencies or infections.)
    * **Surgeon's Query:** {surgeon_query} (The surgeon's specific question or
    concern about the surgical error and potential infection risks.)

    **Requirements:**

    1. **Accurate Infection Prevention Steps:** Outline the most critical infection
    prevention and control measures required to mitigate the risk of infection
    *specifically* in response to the described surgical error.  Your steps should
    be detailed and actionable, suitable for an operating room setting. Prioritize
    the steps based on urgency and clinical significance. Be specific about the
    timing of each step.


    2. **Risk Assessment:** Assess the risk of infection given the surgical error
    and the patient's history.  Quantify the risk if possible (e.g., low, moderate,
    high). Justify your risk assessment.


    3. **Evidence-Based Rationale:** Support all recommendations with credible
    medical evidence.  Cite relevant guidelines, research, or best practices from
    authoritative sources (e.g., CDC, WHO, professional surgical societies).


    4. **Clear and Concise Language:** Use precise and unambiguous language easily
    understood by a surgeon. Avoid jargon or overly technical terms. Focus on
    practical and immediate actions.


    5. **Prioritization:** Clearly prioritize the infection prevention steps in
    order of importance and urgency.

    **Output Format:**

    * **Infection Prevention Steps (Bulleted list):**  Each step should include a
    detailed description and an explanation of its purpose.
    * **Risk Assessment:** A clear statement of the infection risk (low, moderate, high)
    and the rationale for this assessment.
    * **Evidence-Based Citations:**  Provide relevant references for all guidelines
    and recommendations used.


    **Example:**

    (Assume a specific surgical error and patient history are provided)


    **Do not provide any additional information not explicitly requested.
    Only provide the requested response components.**

  """
  prompt: str = instructions.format(
      surgeon_query=surgeon_query,
      patient_history=patient_history
  )

  response: BaseMessage = model.invoke(prompt)
  return {"complication_response": response.content}

