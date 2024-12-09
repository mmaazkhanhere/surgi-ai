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

def anesthesia_consultant_node(state: PreSurgeryState):
    """LangGraph node that gives insights related to anesthesia in surgery"""

    print('Anesthesia Consultant Node')
    patient_history: str = state['patient_history']
    operation: str = state.get('operation')

    instructions = """
        You are an expert anesthesiologist reviewing a surgical procedure and creating guidelines for anesthesia.  Given the patient's history and the planned operation, provide detailed anesthesia recommendations and potential complications.  Your output should be formatted as a structured report, including the following sections:

        **1. Patient History Review:** Summarize relevant patient history, specifically focusing on any pre-existing conditions that might impact anesthesia (e.g., cardiovascular disease, respiratory issues, allergies, previous surgeries).
        **2. Anesthesia Plan:**  Detail the proposed anesthesia technique, including the type of anesthesia (general, regional, local), any premedication, monitoring plan (e.g., EKG, pulse oximetry, capnography), and anticipated anesthetic agents.  Justify your choices based on the patient history and surgical procedure.
        **3. Potential Complications and Management:** List potential complications related to anesthesia, such as difficult airway management, hypotension, and allergic reactions.  For each complication, provide strategies for prevention and management.
        **4. Postoperative Care Considerations:** Outline any specific postoperative care instructions related to anesthesia, such as pain management, respiratory support, and monitoring for delayed reactions.
        **5. Report Writing Guidelines:**  Offer specific phrases and key points that will help in generating a detailed surgical procedure report related to anesthesia operation. 

        **Input Data:**

        * **Patient History:** {patient_history}
        * **Operation:** {operation}


        **Output Format:**

        Use clear and concise language.  Must be less than one paragaph. Be precise and provide specific details to support your recommendations.  Quantify risks whenever possible.  Prioritize patient safety and well-being.
    """

    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
    )

    response: BaseMessage = model.invoke(prompt)
    state["anesthesia_consultant_report"] = response.content
    return state
