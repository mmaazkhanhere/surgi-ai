import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults

from helper_functions.pinecone_vector_store import pinecone_vector_store
from helper_functions.pinecone_vector_store import embeddings

load_dotenv()

#Initialize the Language Learning Model (LLM)
llm_model = ChatGroq(
    model='llama3-70b-8192',
    verbose=True,
    temperature=0.5,
    api_key=os.getenv('GROQ_API_KEY')
)

# llm_model = ChatGoogleGenerativeAI(
#             model='gemini-1.5-flash',
#             verbose=True,
#             temperature=0.5,
#             api_key=os.getenv('GOOGLE_API_KEY')
#         )

@tool
def query_pinecone(surgeon_query: str):
    "Query pinecone database and retreive relevant information based on the query"

    vector_store = pinecone_vector_store()
    embedding = embeddings()

    knowledge = vector_store.from_existing_index(index_name="surgical-assistant",
                                                embedding=embedding)

    qa = RetrievalQA.from_chain_type(llm=llm_model,
                                    chain_type="stuff",
                                    retriever=knowledge.as_retriever()
                                )
    result = qa.invoke(surgeon_query).get("result")
    return result


tavily_search  = TavilySearchResults(max_results=1)    


def pre_surgery_report_crew(
    surgery_name: str,
    patient_age: str,
    prescription_text: str,
    lab_report_text: str,
    scans_text: str
) -> str:

    """A functiont takes 5 inputs and generate a detailed pre surgery report containing various instructions and guidance to help
    the surgeon during surgery"""

    # Defining agents of the crew with updated, precise goals and backstories
    medications_and_prescriptions_summary_agent = Agent(
        llm=llm_model,
        role="Certified Prescription Specialist",
        goal="Summarize current prescriptions, identify drug interactions, and recommend necessary adjustments before {surgery_name}.",
        backstory="Analyzes the patient's prescriptions: {prescription_text} to highlight medications affecting {surgery_name} or anesthesia.",
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )

    test_results_analysis_agent = Agent(
        llm=llm_model,
        role="Licensed Lab Results Specialist",
        goal="Analyze lab and scan reports to identify abnormal findings or risks for {surgery_name}.",
        backstory="Interprets test and scan data: {lab_report_text}, {scans_text} for patient age {patient_age} relevant to {surgery_name}.",
        verbose=True,
        allow_delegation=False,
    )

    anesthesia_plan_advisor_agent = Agent(
        llm=llm_model,
        role="Certified Anesthesia Consultant",
        goal="Create an anesthesia plan based on age {patient_age}, medications {prescription_text}, and test results for {surgery_name}.",
        backstory="Develops an anesthesia strategy considering age, prescriptions, and test/scan findings: {lab_report_text}, {scans_text} for {surgery_name}.",
        verbose=True,
        allow_delegation=False,
    )

    patient_specific_precaution_agent = Agent(
        llm=llm_model,
        role="Certified Pre-Surgery Precautions Analyst",
        goal="Recommend precautions based on age {patient_age}, prescriptions {prescription_text}, and test/scan results for {surgery_name}.",
        backstory="Evaluates patient-specific precautions by analyzing age, medications, and test/scan data: {lab_report_text}, {scans_text} for {surgery_name}.",
        verbose=True,
        allow_delegation=False,
    )

    surgical_risk_analysis_agent = Agent(
        llm=llm_model,
        role="Licensed Surgical Risk Mitigation Expert",
        goal="Assess surgical risks by analyzing age {patient_age}, prescriptions {prescription_text}, and test/scan abnormalities for {surgery_name}.",
        backstory="Identifies risks like bleeding or infection by reviewing age, medications, and test/scan results: {lab_report_text}, {scans_text} for {surgery_name}.",
        verbose=True,
        allow_delegation=False,
        tools=[query_pinecone]
    )

    surgical_instruments_advisor_agent = Agent(
        llm=llm_model,
        role="Surgical Instruments Advisor",
        goal="Recommend appropriate surgical instruments for {surgery_name} based on patient specifics and surgical requirements.",
        backstory="Selects and recommends the best instruments tailored to the patient's condition and the specifics of {surgery_name}.",
        verbose=True,
        allow_delegation=False,
        tools=[query_pinecone]
    )

    surgical_technique_consultant_agent = Agent(
        llm=llm_model,
        role="Surgical Technique Consultant",
        goal="Provide detailed surgical techniques and methodologies for {surgery_name} tailored to the patient's condition.",
        backstory="Develops and outlines optimal surgical techniques for {surgery_name}, considering patient age, health status, and test results.",
        verbose=True,
        allow_delegation=False,
        tools=[query_pinecone]
    )

    complication_forecaster_agent = Agent(
        llm=llm_model,
        role="Complication Forecaster",
        goal="Identify potential complications during {surgery_name} and propose mitigation strategies.",
        backstory="Predicts possible intraoperative and postoperative complications for {surgery_name} based on patient data and surgical parameters.",
        verbose=True,
        allow_delegation=False,
    )

    step_by_step_process_agent = Agent(
        llm=llm_model,
        role="Surgical Procedure Planner",
        goal="Outline a detailed step-by-step process for performing {surgery_name}.",
        backstory="Creates a comprehensive, sequential guide for {surgery_name} to ensure thorough and safe execution.",
        verbose=True,
        allow_delegation=False,
        tools=[query_pinecone]
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
        allow_delegation=False,
        tools=[query_pinecone]
    )

    # Defining tasks for each agent in the pre-surgical report system
    medications_and_prescriptions_summary_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {prescription_text}.\n"
            "2. Pass the relevant data to medications_and_prescriptions_summary_agent.\n"
            "3. Summarize the current prescriptions, identify any potential drug interactions.\n"
            "4. Recommend adjustments or discontinuations necessary before the surgery."
        ),
        expected_output=(
            "Provide a concise summary of current prescriptions, identify potential drug interactions, "
            "and recommend necessary adjustments or discontinuations before {surgery_name}."
        ),
        agent=medications_and_prescriptions_summary_agent,
        tools=[tavily_search]
    )

    test_results_analysis_task = Task(
        description=(
            "1. Receive input: surgery_name, {patient_age}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to test_results_analysis_agent.\n"
            "3. Analyze the lab and scan reports to identify abnormal findings.\n"
            "4. Highlight indicators that may pose risks during the surgery."
        ),
        expected_output=(
            "Provide an analysis of lab and scan reports, identifying abnormal findings or risks for {surgery_name}."
        ),
        agent=test_results_analysis_agent,
    )

    anesthesia_plan_advisor_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {prescription_text}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to anesthesia_plan_advisor_agent.\n"
            "3. Develop a tailored anesthesia plan considering the patient's age and current medications.\n"
            "4. Incorporate relevant findings from test and scan reports to ensure safety and efficacy."
        ),
        expected_output=(
            "Create a tailored anesthesia plan based on age {patient_age}, medications {prescription_text}, "
            "and test/scan findings for {surgery_name}."
        ),
        agent=anesthesia_plan_advisor_agent,
    )

    patient_specific_precaution_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {prescription_text}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to patient_specific_precaution_agent.\n"
            "3. Identify patient-specific precautions based on age, medications, and test/scan results.\n"
            "4. Recommend actionable steps to minimize potential complications during surgery."
        ),
        expected_output=(
            "Recommend specific precautions based on age {patient_age}, medications {prescription_text}, "
            "and test/scan results for {surgery_name}."
        ),
        agent=patient_specific_precaution_agent,
    )

    surgical_risk_analysis_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {prescription_text}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to surgical_risk_analysis_agent.\n"
            "3. Assess potential surgical risks such as bleeding, infection, or other complications.\n"
            "4. Analyze the patient's age, medications, and abnormal test/scan findings to identify risks.\n"
            "5. Offer strategies to mitigate identified risks, enhancing surgical safety."
        ),
        expected_output=(
            "Assess and communicate potential surgical risks by analyzing age {patient_age}, "
            "prescriptions {prescription_text}, and test/scan abnormalities for {surgery_name}."
        ),
        agent=surgical_risk_analysis_agent,
        tools=[query_pinecone]
    )

    surgical_instruments_advisor_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to surgical_instruments_advisor_agent.\n"
            "3. Recommend appropriate surgical instruments tailored to the patient's condition and the specifics of {surgery_name}.\n"
            "4. Justify the selection based on best practices and patient safety."
        ),
        expected_output=(
            "Provide a list of recommended surgical instruments for {surgery_name}, tailored to the patient's condition and surgical requirements."
        ),
        agent=surgical_instruments_advisor_agent,
        tools=[query_pinecone]
    )

    surgical_technique_consultant_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to surgical_technique_consultant_agent.\n"
            "3. Develop and outline optimal surgical techniques for {surgery_name}.\n"
            "4. Ensure techniques are suitable for the patient's age and health status."
        ),
        expected_output=(
            "Provide detailed surgical techniques and methodologies for {surgery_name} tailored to the patient's condition."
        ),
        agent=surgical_technique_consultant_agent,
        tools=[query_pinecone]
    )

    complication_forecaster_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {prescription_text}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to complication_forecaster_agent.\n"
            "3. Identify potential intraoperative and postoperative complications for {surgery_name}.\n"
            "4. Propose strategies to mitigate these complications."
        ),
        expected_output=(
            "Identify potential complications during and after {surgery_name} and propose mitigation strategies."
        ),
        agent=complication_forecaster_agent,
    )

    step_by_step_process_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {lab_report_text}, {scans_text}.\n"
            "2. Pass the relevant data to step_by_step_process_agent.\n"
            "3. Outline a comprehensive, sequential guide for performing {surgery_name}.\n"
            "4. Ensure the process includes all necessary steps to enhance surgical safety and effectiveness."
        ),
        expected_output=(
            "Provide a detailed step-by-step process for performing {surgery_name}, ensuring thorough and safe execution."
        ),
        agent=step_by_step_process_agent,
        tools=[query_pinecone]
    )

    # Defining the compilation task handled by the chief surgeon agent
    chief_surgeon_compilation_task = Task(
        description=(
            "1. Receive input: {surgery_name}, {patient_age}, {prescription_text}, {lab_report_text}, {scans_text}.\n"
            "2. Delegate tasks to all specialized agents.\n"
            "3. Collect outputs from each agent.\n"
            "4. Synthesize the collected information into a comprehensive pre-surgery report, including:\n"
            "5. Iterate with agents for any clarifications or additional data if required.\n"
            "6. Finalize and return the pre-surgery report."
        ),
        expected_output=(
            "Compile a comprehensive pre-surgery report for {surgery_name} by integrating insights from all specialized agents, including instruments, techniques, risks, complications, and procedural steps."
        ),
        agent=chief_surgeon_agent,
        tools=[query_pinecone]
    )

    # Defining the crew with the chief surgeon agent as the manager
    surgical_crew = Crew(
        agents=[
            chief_surgeon_agent,  # Manager agent
            medications_and_prescriptions_summary_agent,
            test_results_analysis_agent,
            anesthesia_plan_advisor_agent,
            patient_specific_precaution_agent,
            surgical_risk_analysis_agent,
            surgical_instruments_advisor_agent,
            surgical_technique_consultant_agent,
            complication_forecaster_agent,
            step_by_step_process_agent
        ],
        tasks=[
            medications_and_prescriptions_summary_task,
            test_results_analysis_task,
            anesthesia_plan_advisor_task,
            patient_specific_precaution_task,
            surgical_risk_analysis_task,
            surgical_instruments_advisor_task,
            surgical_technique_consultant_task,
            complication_forecaster_task,
            step_by_step_process_task,
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