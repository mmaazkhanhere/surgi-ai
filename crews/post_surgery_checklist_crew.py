import os
from dotenv import load_dotenv
from typing import List

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

    physical_therapy_advisor_agent = Agent(
        llm=llm_model,
        role="Physical Therapy Advisor",
        goal="Provide instructions on permissible activities and prescribed rehabilitation exercises.",
        backstory=(
            "This agent specializes in postoperative physical therapy. It outlines activity restrictions, recommended exercises, "
            "and guidelines to aid in the patient's physical recovery."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )

    follow_up_coordinator_agent = Agent(
        llm=llm_model,
        role="Follow-Up Coordinator",
        goal="Outline necessary follow-up visits and monitoring schedules.",
        backstory=(
            "This agent manages follow-up care instructions. It schedules follow-up appointments, outlines what to expect during these visits, "
            "and highlights the importance of adhering to the follow-up schedule."
        ),
        verbose=True,
        allow_delegation=False,
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

    patient_education_specialist_agent = Agent(
        llm=llm_model,
        role="Patient Education Specialist",
        goal="Provide comprehensive educational materials and support resources.",
        backstory=(
            "This agent focuses on patient education. It creates educational content that informs the patient about their recovery process, "
            "how to manage their health post-surgery, and available support resources."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )

    discharge_instructions_agent = Agent(
        llm=llm_model,
        role="Discharge Instructions Expert",
        goal="Generate clear and concise discharge instructions for the patient.",
        backstory=(
            "This agent creates detailed discharge instructions, ensuring the patient understands the necessary steps to take upon leaving the hospital, "
            "including medication adherence, activity levels, and follow-up care."
        ),
        verbose=True,
        allow_delegation=False,
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

    physical_therapy_advisor_task = Task(
        description=(
            "1. Provide instructions on permissible activities and prescribed rehabilitation exercises based on surgery details {surgery_details} and patient conditions{patient_conditions}.\n"
            "2. Include guidelines to aid in physical recovery and prevent complications.\n"
            "3. Ensure recommendations are practical and tailored to the patient's recovery needs."
        ),
        expected_output=(
            "A set of activity restrictions and rehabilitation exercises designed to aid in the patient's physical recovery."
        ),
        agent=physical_therapy_advisor_agent,
        tools=[tavily_search]
    )

    follow_up_coordinator_task = Task(
        description=(
            "1. Outline necessary follow-up visits and monitoring schedules based on surgery details {surgery_details} and patient conditions {patient_conditions}.\n"
            "2. Include information on what to expect during these appointments.\n"
            "3. Emphasize the importance of adhering to the follow-up schedule for optimal recovery."
        ),
        expected_output=(
            "A detailed schedule of follow-up appointments with descriptions of what to expect during each visit."
        ),
        agent=follow_up_coordinator_agent,
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

    patient_education_specialist_task = Task(
        description=(
            "1. Provide comprehensive educational materials and support resources based on surgery details {surgery_details} and patient conditions {patient_conditions}.\n"
            "2. Include information on managing health post-surgery, lifestyle modifications, and available support services.\n"
            "3. Ensure educational content is clear, informative, and supportive."
        ),
        expected_output=(
            "Educational materials and resources that inform the patient about their recovery process, health management, and support options."
        ),
        agent=patient_education_specialist_agent, 
        tools=[tavily_search]
    )

    discharge_instructions_task = Task(
        description=(
            "1. Generate clear and concise discharge instructions based on surgery details {surgery_details}, conversation {surgery_conversation}, and patient conditions {patient_conditions}.\n"
            "2. Include necessary steps to take upon leaving the hospital, such as medication adherence, activity levels, and follow-up care.\n"
            "3. Ensure instructions are easy to understand and actionable."
        ),
        expected_output=(
            "A set of discharge instructions outlining the necessary steps and guidelines for the patient upon leaving the hospital."
        ),
        agent=discharge_instructions_agent
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
            physical_therapy_advisor_agent,
            follow_up_coordinator_agent,
            complications_monitor_agent,
            dietary_consultant_agent,
            patient_education_specialist_agent,
            discharge_instructions_agent,
            final_checklist_compiler_agent
        ],
        tasks=[
            checklist_manager_task,
            wound_care_task,
            medication_manager_task,
            physical_therapy_advisor_task,
            follow_up_coordinator_task,
            complications_monitor_task,
            dietary_consultant_task,
            patient_education_specialist_task,
            discharge_instructions_task,
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
