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

def instrumentation_decider_node(state: PreSurgeryState):
    """LangGraph node that gives insights on instruments to used based on the surgery"""

    print('Instrumentation Node')
    patient_history: str = state['patient_history']
    operation: str = state.get('operation')

    instructions = """
        You are an expert in surgical instrumentation, with a focus on the precise selection of surgical instruments to enhance the surgical procedure.
        Analyze the provided patient history and the planned operation to determine the most suitable surgical instruments.  Consider potential complications and the patient's specific needs.  Provide valuable insights and recommendations for instrument selection, including justifications for your choices.

        **Patient History:** {patient_history}

        **Operation:** {operation}

        **Guidelines for Output:**

        1. **Instrument List:** Provide a list of essential surgical instruments, categorized by their purpose (e.g., dissection, retraction, hemostasis, closure).

        2. **Justification:**  For each instrument, explain why it is necessary, considering the patient's history and the specifics of the operation. Mention any relevant anatomical considerations or potential complications that the instrument addresses.

        3. **Alternatives:** If applicable, suggest alternative instruments and explain the rationale for choosing the primary instrument over the alternatives.

        4. **Report Integration:** Explain how the chosen instrument set can be used to enhance the surgical procedure report. Suggest specific details and data points that should be included in the report about the use of each instrument to enhance clarity.

        5. **Format:**  The output should be structured as a bulleted list. Each bullet point should include the instrument name, a brief description of its function, the rationale for its use in this specific case, and notes about its incorporation into the surgical report. Must be less than one paragraph


        **Example:**

        * **Instrument:**  Monopolar Electrosurgical Device
            * **Function:**  Used for tissue dissection and hemostasis.
            * **Rationale:** In a laparoscopic cholecystectomy, the monopolar electrosurgical device is essential for precise dissection of the cystic duct and artery, minimizing bleeding.  The patient's history of previous right upper quadrant surgery may lead to adhesions, necessitating a more cautious dissection approach with fine control of the electrosurgical device.
            * **Report Integration:**  The surgical report should detail the settings used for the electrosurgical device during cystic duct and artery division, and note any changes in settings, to demonstrate a meticulous approach to hemostasis.  

            **Instrument:** .... (add more relevant instruments)
        **Output:**

    """
    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
    )

    response: BaseMessage = model.invoke(prompt)
    state["instrumentation_report"] = response.content
    return state
