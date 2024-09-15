import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm_model = ChatGroq(
    model='llama3-70b-8192',
    verbose=True,
    temperature=0.5,
    api_key=os.getenv('GROQ_API_KEY')
)
tavily_search  = TavilySearchResults(max_results=1)


def post_surgery_checklist_crew(surgery_details: str, surgery_conversation: str, patient_conditions: str) -> str:
    """
    Crew of agents responsible for generating a comprehensive post-surgery checklist.
    """

    # Agent Definitions

    checklist_manager_agent = Agent(
        llm=llm_model,
        role="Post-Surgery Checklist Manager",
        goal="Coordinate the creation and compilation of the post-surgery checklist by integrating inputs from specialized agents.",
        backstory=(
            "This agent oversees the generation of the post-surgery checklist. It ensures that all necessary tasks are delegated "
            "to specialized agents and that their outputs are compiled into a cohesive and comprehensive checklist."
        ),
        verbose=True,
        allow_delegation=True
    )

    wound_care_agent = Agent(
        llm=llm_model,
        role="Wound Care Specialist",
        goal="Generate detailed wound care instructions based on surgery details.",
        backstory=(
            "This agent specializes in postoperative wound care. It creates detailed instructions for caring for surgical wounds, "
            "including cleaning, dressing changes, and signs of infection to monitor."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )

    medication_manager_agent = Agent(
        llm=llm_model,
        role="Medication Manager",
        goal="Document medications to be taken post-surgery, including dosages and administration times.",
        backstory=(
            "This agent focuses on medication management post-surgery. It provides a detailed list of prescribed medications, "
            "dosages, administration schedules, and potential side effects."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )


    complications_monitor_agent = Agent(
        llm=llm_model,
        role="Complications Monitor",
        goal="Provide guidelines on signs and symptoms that require immediate attention.",
        backstory=(
            "This agent focuses on monitoring potential complications post-surgery. It lists signs and symptoms that the patient should watch for "
            "and instructs when to seek immediate medical attention."
        ),
        verbose=True,
        allow_delegation=False,
    )

    dietary_consultant_agent = Agent(
        llm=llm_model,
        role="Dietary Consultant",
        goal="Offer nutritional recommendations and restrictions based on surgery type.",
        backstory=(
            "This agent specializes in postoperative nutrition. It provides dietary guidelines, including foods to eat, foods to avoid, "
            "and nutritional supplements that may aid in recovery."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )


    final_checklist_compiler_agent = Agent(
        llm=llm_model,
        role="Final Checklist Compiler",
        goal="Integrate inputs from all specialized agents to compile the final comprehensive post-surgery checklist.",
        backstory=(
            "This agent consolidates the contributions from all specialized agents to create a well-structured and comprehensive post-surgery checklist. "
            "It ensures consistency, clarity, and thoroughness in the final document."
        ),
        verbose=True,
        allow_delegation=False,
    )

    # Task Definitions

    checklist_manager_task = Task(
        description=(
            "1. Initiate the post-surgery checklist generation process using the provided surgery details {surgery_details}, conversation {surgery_conversation}, and patient conditions {patient_conditions}.\n"
            "2. Delegate specific checklist categories to the specialized agents: \n"
            "3. Collect the generated checklist items from each specialized agent.\n"
            "4. Ensure all checklist items are comprehensive and tailored to the specific surgery and patient condition.\n"
            "5. Pass the collected checklist items to the Final Checklist Compiler for integration."
        ),
        expected_output=(
            "A structured compilation of checklist items from specialized agents, ready to be integrated into the final document."
        ),
        agent=checklist_manager_agent,
    )

    wound_care_task = Task(
        description=(
            "1. Generate detailed wound care instructions based on the provided surgery details {surgery_details} and patient conditions {patient_conditions}.\n"
            "2. Include information on cleaning, dressing changes, and signs of infection to monitor.\n"
            "3. Ensure instructions are clear, actionable, and tailored to the specific surgical procedure."
        ),
        expected_output=(
            "A comprehensive set of wound care instructions tailored to the specific surgery and patient condition {patient_conditions}."
        ),
        agent=wound_care_agent,
        tools=[tavily_search]
    )

    medication_manager_task = Task(
        description=(
            "1. Document all medications prescribed post-surgery, including dosages and administration times, based on surgery details {surgery_details} and patient conditions {patient_conditions}.\n"
            "2. Include information on how to manage potential side effects.\n"
            "3. Ensure accuracy and clarity in medication instructions."
        ),
        expected_output=(
            "A detailed list of prescribed medications with dosages, administration schedules, and management of potential side effects."
        ),
        agent=medication_manager_agent,
        tools=[tavily_search]
    )


    complications_monitor_task = Task(
        description=(
            "1. Provide guidelines on signs and symptoms that require immediate medical attention based on surgery details {surgery_details} and patient conditions {patient_conditions}.\n"
            "2. Include information on how to recognize and respond to potential complications.\n"
            "3. Ensure the guidelines are clear and actionable."
        ),
        expected_output=(
            "A comprehensive list of signs and symptoms indicating potential complications, along with appropriate responses."
        ),
        agent=complications_monitor_agent
    )

    dietary_consultant_task = Task(
        description=(
            "1. Offer nutritional recommendations and restrictions based on the surgery details {surgery_details} and patient conditions {patient_conditions}.\n"
            "2. Include information on foods to eat, foods to avoid, and any necessary dietary supplements.\n"
            "3. Ensure dietary guidelines support optimal healing and recovery."
        ),
        expected_output=(
            "A set of dietary guidelines including recommended foods, foods to avoid, and nutritional supplements to aid in recovery."
        ),
        agent=dietary_consultant_agent,
        tools=[tavily_search]
    )


    final_checklist_compiler_task = Task(
        description=(
            "1. Gather the checklist items from all specialized agents: \n"
            "2. Integrate these items into a single, cohesive post-surgery checklist.\n"
            "3. Ensure the checklist is organized by category and formatted consistently.\n"
            "4. Review the compiled checklist for accuracy, clarity, and comprehensiveness.\n"
            "5. Finalize the checklist document for distribution or use."
        ),
        expected_output=(
            "A finalized, well-structured post-surgery checklist encompassing all necessary care instructions and guidelines."
        ),
        agent=final_checklist_compiler_agent
    )

    # Defining the Crew

    checklist_crew = Crew(
        agents=[
            checklist_manager_agent,
            wound_care_agent,
            medication_manager_agent,
            complications_monitor_agent,
            dietary_consultant_agent,
            final_checklist_compiler_agent
        ],
        tasks=[
            checklist_manager_task,
            wound_care_task,
            medication_manager_task,
            complications_monitor_task,
            dietary_consultant_task,
            final_checklist_compiler_task
        ],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential
    )

    # Initializing the Crew with necessary inputs

    initial_inputs = {
        'surgery_details': surgery_details,           
        'surgery_conversation': surgery_conversation, 
        'patient_conditions': patient_conditions
    }

    # Executing the Crew

    result = checklist_crew.kickoff(initial_inputs)
    return result
