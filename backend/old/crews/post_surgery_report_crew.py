import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process

from langchain_groq import ChatGroq

load_dotenv()

llm_model = ChatGroq(
    model='llama3-70b-8192',
    verbose=True,
    temperature=0.5,
    api_key=os.getenv('GROQ_API_KEY')
)


def operative_report_crew(surgery_details: str, surgeon_conversation: str, patient_condition: str) -> str:
    """
    Creates a crew of agents responsible for generating comprehensive operative reports.
    The report includes preoperative and postoperative diagnoses, patient condition after surgery, and all medications used during the procedure.
    """

    # defining agents
    report_manager_agent = Agent(
        llm=llm_model,
        role="Operative Report Manager",
        goal="Coordinate the compilation of the operative report by integrating inputs from specialized agents.",
        backstory=(
            "This agent oversees the creation of the operative report. It ensures that all necessary information "
            "is collected from specialized agents and compiled into a cohesive and comprehensive document."
        ),
        verbose=True,
        allow_delegation=True  # Allows delegating tasks to specialized agents
    )


    postoperative_diagnosis_agent = Agent(
        llm=llm_model,
        role="Postoperative Diagnosis Analyst",
        goal="Extract and document postoperative diagnoses based on surgery details and patient condition after surgery.",
        backstory=(
            "This agent focuses on identifying and documenting the patient's postoperative diagnoses. "
            "It reviews surgery details and notes to capture the conditions following the surgery."
        ),
        verbose=True,
        allow_delegation=False,
    )

    patient_condition_agent = Agent(
        llm=llm_model,
        role="Patient Condition Analyst",
        goal="Assess and document the patient's condition after surgery.",
        backstory=(
            "This agent assesses the patient's status post-surgery, documenting vital signs, recovery progress, and any immediate post-operative observations."
        ),
        verbose=True,
        allow_delegation=False,
    )

    medications_agent = Agent(
        llm=llm_model,
        role="Medications Recorder",
        goal="Document all medications administered during the surgical procedure.",
        backstory=(
            "This agent records all medications used in association with the surgical procedure, including dosages and administration times."
        ),
        verbose=True,
        allow_delegation=False,
    )

    final_report_agent = Agent(
        llm=llm_model,
        role="Final Report Compiler",
        goal="Integrate inputs from all specialized agents to compile the final operative report.",
        backstory=(
            "This agent consolidates the analyses from the Preoperative Diagnosis Analyst, Postoperative Diagnosis Analyst, "
            "Patient Condition Analyst, and Medications Recorder to create a comprehensive and coherent operative report. "
            "It ensures that all sections of the report are well-organized and professionally formatted."
        ),
        verbose=True,
        allow_delegation=False,
    )

    # defining tasks
    report_manager_task = Task(
        description=(
            "1. Initiate the operative report generation process using the provided surgery details {surgery_details}, notes, and patient information.\n"
            "2. Delegate specific sections of the report to the specialized agents: \n"
            "3. Collect the outputs from each specialized agent.\n"
            "4. Ensure all sections are comprehensive and accurately reflect the surgery's outcomes and patient care details.\n"
            "5. Pass the collected information to the Final Report Compiler for integration."
        ),
        expected_output=(
            "A structured compilation of inputs from specialized agents, ready to be integrated into the final report."
        ),
        agent=report_manager_agent,
    )


    postoperative_diagnosis_task = Task(
        description=(
            "1. Analyze the surgery details: {surgery_details} and patient condition after surgery: {patient_condition}.\n"
            "2. Identify and document all postoperative diagnoses based on the surgery outcomes.\n"
            "3. Ensure that the diagnoses are accurately captured and clearly stated.\n"
            "4. Provide a detailed summary of postoperative conditions."
        ),
        expected_output=(
            "A comprehensive list of postoperative diagnoses that accurately reflect the patient's conditions following the surgery."
        ),
        agent=postoperative_diagnosis_agent
    )

    patient_condition_task = Task(
        description=(
            "1. Assess the patient's condition after surgery using the provided surgeon conversation during surgery: {surgeon_conversation}.\n"
            "2. Document vital signs, recovery progress, and any immediate post-operative observations.\n"
            "3. Ensure that all aspects of the patient's post-surgery condition are accurately captured.\n"
            "4. Provide a detailed report on the patient's status following the procedure."
        ),
        expected_output=(
            "A detailed account of the patient's condition after surgery, including vital signs, recovery progress, and any immediate observations."
        ),
        agent=patient_condition_agent
    )

    medications_task = Task(
        description=(
            "1. Review the surgery details: {surgery_details} and surgeon conversation: {surgeon_conversation}.\n"
            "2. Identify all medications administered during the surgical procedure, including dosages and administration times.\n"
            "3. Document each medication accurately, ensuring completeness and clarity.\n"
            "4. Provide a comprehensive list of medications used during the surgery."
        ),
        expected_output=(
            "A complete and accurate list of all medications administered during the surgical procedure, including dosages and administration times."
        ),
        agent=medications_agent
    )

    final_report_task = Task(
        description=(
            "1. Gather the analyses from the Preoperative Diagnosis Analyst, Postoperative Diagnosis Analyst, "
            "Patient Condition Analyst, and Medications Recorder.\n"
            "2. Integrate these inputs into a single, cohesive operative report.\n"
            "3. Ensure that the report includes sections on preoperative diagnoses, postoperative diagnoses, "
            "patient condition after surgery, and medications used during the procedure.\n"
            "4. Format the report in a clear, professional manner suitable for medical records and review.\n"
            "5. Verify the accuracy and completeness of the report."
        ),
        expected_output=(
            "A finalized, well-structured operative report encompassing all necessary sections and details."
        ),
        agent=final_report_agent
    )

    # creating crew
    operative_crew = Crew(
        agents=[
            report_manager_agent,
            postoperative_diagnosis_agent,
            patient_condition_agent,
            medications_agent,
            final_report_agent
        ],
        tasks=[
            report_manager_task,
            postoperative_diagnosis_task,
            patient_condition_task,
            medications_task,
            final_report_task
        ],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential  
    )

    # Initializing the Crew with necessary inputs
    initial_inputs = {
        'surgery_details': surgery_details,   
        'surgeon_conversation': surgeon_conversation,      
        'patient_condition': patient_condition          
    }

    # Executing the Crew

    result = operative_crew.kickoff(initial_inputs)
    return result
