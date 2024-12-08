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

def surgical_workflow_node(state: PreSurgeryState):
    """LangGraph node that gives insights based on scans of the patient"""

    print('Surgical Workflow')
    accumulator_output: str = state['accumulator']
    operation: str = state.get('operation')

    instructions = """
        Create a detailed step-by-step surgical procedure report for the following surgery based on the provided information from the accumulator node.  The report should be comprehensive, covering all aspects of the procedure.

        **Input:**
        * Accumulator Node Output: {accumulator_output}

        **Report Structure and Content:**

        1. **Patient Information:** (Briefly summarize relevant patient information from the accumulator node)
        2. **Surgical Procedure:**
            * **Preoperative Preparations:** Describe necessary preparations, including patient positioning, monitoring, and any specific equipment setup.
            * **Step-by-Step Instructions:** Provide a detailed, numbered sequence of surgical steps.  For each step, specify the instruments used, any critical decisions to be made, and potential alternatives.  Use clear, concise language suitable for a surgical team.
            * **Anesthesia Plan:**  Detail the type and method of anesthesia to be used. Outline the monitoring plan and protocols to manage any potential adverse events.
            * **Emergency Protocols:** Include procedures for potential intraoperative emergencies. Clearly define actions to take in case of unexpected complications (e.g., hemorrhage, cardiac arrest). This should include a clear hierarchy of actions and personnel responsible.
            * **Potential Complications and Management:** List potential complications associated with the surgery and the corresponding management strategies.
        3. **Postoperative Care:** (Summarize key aspects of post-operative care from the accumulator node)
        4. **Risks and Benefits:** (Summarize from accumulator node)


        **Instructions:**

        * The report must be detailed and comprehensive, providing step-by-step guidance for the surgical team.
        * Clearly identify the instruments to be used in each step.
        * Address potential risks associated with the surgery.
        * Detail how to use anesthesia.
        * Outline protocols for potential emergencies during the surgery.
        * Clearly explain potential complications that may arise and how to manage them.
        * The response should be well-organized and easy to read.
        * Use headings and subheadings to structure the report.

        **Example:**

        **(Note: This is a placeholder example and does not represent a complete surgical procedure report.)**

        1. **Patient Information:** A 50-year-old female with history of hypertension.

        2. **Surgical Procedure:**
            * **Preoperative Preparations:** Standard monitoring, etc.
            * **Step-by-Step Instructions:**
                1. Incision: Scalpel.
                2. Dissection: Forceps.
                3. ...
            * **Anesthesia Plan:** General anesthesia.
            * **Emergency Protocols:** ...
            * **Potential Complications and Management:** ...
        3. **Postoperative Care:** ...
        4. **Risks and Benefits:** ...

        **Generate Report:**
    """
    prompt: str = instructions.format(
        operation=operation,
        accumulator_output=accumulator_output
    )

    response: BaseMessage = model.invoke(prompt)
    state["surgical_workflow_report"] = response.content
    return state
