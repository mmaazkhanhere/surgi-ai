import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

from langchain_groq import ChatGroq

load_dotenv()

# Initialize the Language Learning Model (LLM)
llm_model = ChatGroq(
    model='llama3-70b-8192',
    verbose=True,
    temperature=0.5,
    api_key=os.getenv('GROQ_API_KEY')
)

def pre_surgery_crew(
    surgery_name: str,
    patient_age: str,
    prescription_text: str,
    lab_report_text: str,
    scans_text: str
) -> str:

    # Defining consolidated agents of the crew with updated, precise goals and backstories

    medications_and_prescriptions_summary_agent = Agent(
        llm=llm_model,
        role="Certified Prescription Specialist",
        goal="Summarize current prescriptions, identify drug interactions, and recommend necessary adjustments before {surgery_name}.",
        backstory="Analyzes the patient's prescriptions: {prescription_text} to highlight medications affecting {surgery_name} or anesthesia.",
        verbose=True,
        allow_delegation=False,
    )

    test_results_analysis_agent = Agent(
        llm=llm_model,
        role="Licensed Lab Results Specialist",
        goal="Analyze lab and scan reports to identify abnormal findings or risks for {surgery_name}.",
        backstory="Interprets test and scan data: {lab_report_text}, {scans_text} for patient age {patient_age} relevant to {surgery_name}.",
        verbose=True,
        allow_delegation=False,
    )

    anesthesia_and_precautions_advisor_agent = Agent(
        llm=llm_model,
        role="Certified Anesthesia and Precautions Consultant",
        goal=(
            "Create an anesthesia plan and recommend patient-specific precautions based on age {patient_age}, "
            "medications {prescription_text}, and test results for {surgery_name}."
        ),
        backstory=(
            "Develops an anesthesia strategy and evaluates patient-specific precautions considering age, "
            "prescriptions, and test/scan findings: {lab_report_text}, {scans_text} for {surgery_name}."
        ),
        verbose=True,
        allow_delegation=False,
    )

    surgical_risk_and_instruments_expert_agent = Agent(
        llm=llm_model,
        role="Licensed Surgical Risk and Instruments Expert",
        goal=(
            "Assess surgical risks and provide a comprehensive list of instruments required for {surgery_name}, "
            "ensuring all necessary tools are available and appropriate for the procedure."
        ),
        backstory=(
            "Identifies risks like bleeding or infection by reviewing age, medications, and test/scan results: "
            "{lab_report_text}, {scans_text} for {surgery_name}. Additionally, compiles a detailed list of "
            "surgical instruments needed for the procedure."
        ),
        verbose=True,
        allow_delegation=False,
    )

    chief_surgeon_agent = Agent(
        llm=llm_model,
        role="Chief Surgeon Advisor",
        goal=(
            "Orchestrate the generation of a comprehensive pre-surgery report for {surgery_name} by delegating tasks to specialized agents, "
            "collecting their responses, and compiling the final report."
        ),
        backstory=(
            "Acts as the head of the surgical crew, managing and coordinating with specialized agents to gather all necessary information "
            "and compile a detailed pre-surgery report for {surgery_name}."
        ),
        verbose=True,
        allow_delegation=True,  # Allows the chief surgeon agent to delegate tasks
    )

    # Defining tasks for each agent in the pre-surgical report system

    medications_and_prescriptions_summary_task = Task(
        description=(
            "1. Summarize the current prescriptions, identify any potential drug interactions.\n"
            "2. Recommend adjustments or discontinuations necessary before the surgery."
        ),
        expected_output=(
            "Provide a concise summary of current prescriptions, identify potential drug interactions, "
            "and recommend necessary adjustments or discontinuations before {surgery_name}."
        ),
        agent=medications_and_prescriptions_summary_agent,
    )

    test_results_analysis_task = Task(
        description=(
            "1. Analyze the lab and scan reports to identify abnormal findings.\n"
            "2. Highlight indicators that may pose risks during the surgery."
        ),
        expected_output=(
            "Provide an analysis of lab and scan reports, identifying abnormal findings or risks for {surgery_name}."
        ),
        agent=test_results_analysis_agent,
    )

    anesthesia_and_precautions_advisor_task = Task(
        description=(
            "1. Develop a tailored anesthesia plan considering the patient's age and current medications.\n"
            "2. Recommend patient-specific precautions based on test and scan reports to ensure safety and efficacy."
        ),
        expected_output=(
            "Create a tailored anesthesia plan based on age {patient_age}, medications {prescription_text}, "
            "and test/scan findings for {surgery_name}. Recommend specific precautions to minimize potential complications."
        ),
        agent=anesthesia_and_precautions_advisor_agent,
    )

    surgical_risk_and_instruments_expert_task = Task(
        description=(
            "1. Assess potential surgical risks such as bleeding, infection, or other complications.\n"
            "2. Analyze the patient's age, medications, and abnormal test/scan findings to identify risks.\n"
            "3. Provide a comprehensive list of surgical instruments required for {surgery_name}, including specifications and availability."
        ),
        expected_output=(
            "Assess and communicate potential surgical risks by analyzing age {patient_age}, "
            "prescriptions {prescription_text}, and test/scan abnormalities for {surgery_name}. "
            "Provide a detailed list of surgical instruments required, including specifications and availability."
        ),
        agent=surgical_risk_and_instruments_expert_agent,
    )

    # Defining the compilation task handled by the chief surgeon agent
    chief_surgeon_compilation_task = Task(
        description=(
            "1. Receive input: surgery_name: {surgery_name}, patient_age: {patient_age}, prescription_text: {prescription_text}, "
            "lab_report_text: {lab_report_text}, scans_text: {scans_text}.\n"
            "2. Delegate tasks to all specialized agents.\n"
            "3. Collect outputs from each agent.\n"
            "4. Synthesize the collected information into a comprehensive surgical guideline.\n"
            "5. Iterate with agents for any clarifications or additional data if required.\n"
            "6. Finalize and return the pre-surgery report.\n"
            "7. Ensure that each agent is queried only once to prevent redundancy."
        ),
        expected_output=(
            "Compile a comprehensive pre-surgery report for {surgery_name} by integrating insights from all specialized agents."
        ),
        agent=chief_surgeon_agent,
    )

    # Defining the crew with the chief surgeon agent as the manager
    surgical_crew = Crew(
        agents=[
            medications_and_prescriptions_summary_agent,
            test_results_analysis_agent,
            anesthesia_and_precautions_advisor_agent,
            surgical_risk_and_instruments_expert_agent,
            chief_surgeon_agent  # Manager agent
        ],
        tasks=[
            medications_and_prescriptions_summary_task,
            test_results_analysis_task,
            anesthesia_and_precautions_advisor_task,
            surgical_risk_and_instruments_expert_task,
            chief_surgeon_compilation_task  # Only the compilation task is initiated by the chief surgeon
        ],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential  # Adjust if Process.interactive is supported
    )

    # Initiate the crew with all necessary inputs
    result = surgical_crew.kickoff({
        'surgery_name': surgery_name,
        'patient_age': patient_age,
        'prescription_text': prescription_text,
        'lab_report_text': lab_report_text,
        'scans_text': scans_text
    })
    
    return result
