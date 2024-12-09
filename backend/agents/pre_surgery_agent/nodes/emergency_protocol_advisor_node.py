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

def emergency_protocol_advisor_node(state: PreSurgeryState):
    """LangGraph node that gives insights related to potential emergency in surgery"""

    print('Emergency Protocol Advisor Node')
    patient_history: str = state['patient_history']
    operation: str = state.get('operation')

    instructions = """
        Act as an emergency protocol expert.  Given a patient's history and a planned surgical procedure, provide potential emergency guidelines and critical considerations.

        **Input:**

        * **Patient History:** [Detailed patient history, including relevant medical conditions, allergies, past surgeries, and current medications.]
        * **Operation:** [Specific surgical procedure to be performed.]

        **Guidelines for the AI Agent:**

        1. **Identify Potential Emergencies:** Based on the patient history and the operation, list potential medical emergencies that could arise during the procedure (e.g., hemorrhage, anaphylaxis, cardiac arrest).
        2. **Emergency Protocols:** For each identified potential emergency, provide detailed emergency guidelines. This includes immediate actions to be taken, necessary equipment, personnel roles, and crucial monitoring parameters.
        3. **Patient-Specific Considerations:** Tailor the guidelines to the specific patient, taking into account their medical history, current health status, and any previous adverse reactions or complications.  If there are contraindications or special considerations based on the patient's history, be sure to mention them specifically. For example, if the patient has a bleeding disorder, hemorrhage protocols should be emphasized. If they have a history of allergic reactions to specific anesthetics, then the protocols for anaphylaxis must be explicitly mentioned.
        4. **Report Writing Assistance:**  Format your output as concise bullet points for easy integration into a surgical procedure report. Each bullet point should describe a potential emergency, followed by the steps required to mitigate that emergency.

        **Output Format:**

        Use the following format for each potential emergency:

        * **Potential Emergency:** [Clearly state the potential emergency]
        * **Immediate Actions:** [List the immediate actions to be taken.]
        * **Equipment:** [List the necessary equipment.]
        * **Personnel Roles:** [Outline the responsibilities of the surgical team members.]
        * **Monitoring Parameters:** [Specify critical parameters to be monitored.]
        * **Patient-Specific Considerations:** [Address any patient-specific risk factors and adjustments to standard protocols.]
        * **Output should be less than 1 paragraphs


        **Example:**

        * **Potential Emergency:** Hemorrhage
        * **Immediate Actions:** Apply direct pressure to the bleeding site, prepare for IV fluid bolus, notify anesthesia.
        * **Equipment:** Hemostatic agents, surgical suction, IV fluids.
        * **Personnel Roles:** Surgeon controls bleeding, circulating nurse manages IV fluids, anesthesiologist maintains hemodynamic stability.
        * **Monitoring Parameters:** Heart rate, blood pressure, oxygen saturation.
        * **Patient-Specific Considerations:** Patient has history of bleeding disorder; administer Factor VIIa (or similar) as per protocol.

        ..... (continue giving other potential emergencies and guidelines)


    """

    prompt: str = instructions.format(
        operation=operation,
        patient_history=patient_history,
    )

    response: BaseMessage = model.invoke(prompt)
    state["emergency_protocol_advicer"] = response.content
    return state
