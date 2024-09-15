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

tavily_search  = TavilySearchResults(max_results=2)

def surgery_post_faq_crew(surgery_details: str, surgery_conversation: str) -> str:
    """
    Creates a crew of agents responsible for generating comprehensive post-surgery FAQs.
    The FAQs cover postoperative care, risks and complications, medications, recovery processes, patient condition, and general post-surgery information.

    Returns:
        str: A comprehensive post-surgery FAQ document.
    """

    # Agent Definitions

    faq_manager_agent = Agent(
        llm=llm_model,
        role="Post-Surgery FAQ Manager",
        goal="Coordinate the creation and compilation of post-surgery FAQs by integrating inputs from specialized agents.",
        backstory=(
            "This agent oversees the generation of post-surgery FAQs. It ensures that all necessary questions and answers "
            "are collected from specialized agents and compiled into a cohesive and comprehensive FAQ document."
        ),
        verbose=True,
        allow_delegation=True  # Allows delegating tasks to specialized agents
    )

    postoperative_care_agent = Agent(
        llm=llm_model,
        role="Postoperative Care Specialist",
        goal="Generate and answer FAQs about post-surgery recovery and aftercare.",
        backstory=(
            "This agent specializes in postoperative care, creating questions and answers regarding recovery processes, wound care, "
            "rehabilitation exercises, and signs of complications. It utilizes surgery details and surgeon conversations to tailor the FAQs to specific scenarios encountered during the procedure."
        ),
        verbose=True,
        allow_delegation=False,
    )

    risks_complications_agent = Agent(
        llm=llm_model,
        role="Risks and Complications Analyst",
        goal="Generate and answer FAQs about the risks and potential complications associated with surgeries.",
        backstory=(
            "This agent focuses on identifying and explaining the risks and possible complications that may arise after surgery, "
            "providing clear and informative responses to common concerns. It draws insights from surgery details and conversations to address real-world issues encountered during the procedure."
        ),
        verbose=True,
        allow_delegation=False,
    )

    medications_agent = Agent(
        llm=llm_model,
        role="Medications Specialist",
        goal="Generate and answer FAQs about medications used during and after surgical procedures.",
        backstory=(
            "This agent specializes in medications related to surgery, including pain management and preventive antibiotics. "
            "It formulates questions and provides detailed answers about medication types, dosages, administration times, and potential side effects, incorporating specifics from the surgery details and surgeon's notes."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[tavily_search]
    )

    recovery_process_agent = Agent(
        llm=llm_model,
        role="Recovery Process Advisor",
        goal="Generate and answer FAQs about the recovery and rehabilitation process following surgery.",
        backstory=(
            "This agent focuses on the recovery phase post-surgery, creating questions and answers about timelines, physical therapy, lifestyle adjustments, "
            "and strategies to promote optimal healing. It leverages surgery details and surgeon conversations to provide realistic and practical guidance."
        ),
        verbose=True,
        allow_delegation=False,
    )

    patient_condition_agent = Agent(
        llm=llm_model,
        role="Patient Condition Analyst",
        goal="Generate and answer FAQs about assessing and documenting patient condition after surgery.",
        backstory=(
            "This agent specializes in assessing patient status post-surgery, providing questions and answers related to vital signs, recovery progress, "
            "and immediate post-operative observations. It uses surgery details and surgeon conversations to ensure the FAQs address actual patient conditions encountered."
        ),
        verbose=True,
        allow_delegation=False,
    )

    general_information_agent = Agent(
        llm=llm_model,
        role="General Post-Surgery Information Expert",
        goal="Generate and answer general FAQs about post-surgery practices, protocols, and advancements.",
        backstory=(
            "This agent provides broad information about post-surgery practices, including standard protocols, recent advancements in post-operative care, "
            "and best practices to ensure patient safety and successful recovery. It incorporates insights from surgery details and surgeon conversations to keep the information relevant and up-to-date."
        ),
        verbose=True,
        allow_delegation=False,
    )

    final_faq_compiler_agent = Agent(
        llm=llm_model,
        role="Final FAQ Compiler",
        goal="Integrate inputs from all specialized agents to compile the final comprehensive FAQ document.",
        backstory=(
            "This agent consolidates the questions and answers from all specialized agents to create a well-structured and comprehensive FAQ document. "
            "It ensures consistency, clarity, and professionalism in the final compilation, tailoring the content based on the provided surgery details and conversation."
        ),
        verbose=True,
        allow_delegation=False,
    )

    # Task Definitions

    faq_manager_task = Task(
        description=(
            "1. Initiate the post-surgery FAQ generation process using the provided surgery details {surgery_details} and surgery conversation {surgery_conversation}.\n"
            "2. Delegate specific FAQ categories to the specialized agents: \n"
            "3. Collect the generated FAQs from each specialized agent.\n"
            "4. Ensure all FAQs are comprehensive and cover all necessary aspects of post-surgery care and information.\n"
            "5. Pass the collected FAQs to the Final FAQ Compiler for integration."
        ),
        expected_output=(
            "A structured compilation of FAQs from specialized agents, ready to be integrated into the final document."
        ),
        agent=faq_manager_agent,
    )

    postoperative_care_task = Task(
        description=(
            "1. Generate a list of common questions related to postoperative care, utilizing the provided surgery details {surgery_details} and surgery conversation {surgery_conversation}.\n"
            "2. Provide detailed answers for each question, ensuring clarity and accuracy.\n"
            "3. Cover topics like recovery timelines, wound care, rehabilitation exercises, and signs of complications."
        ),
        expected_output=(
            "A set of well-formulated questions and answers pertaining to postoperative care."
        ),
        agent=postoperative_care_agent
    )

    risks_complications_task = Task(
        description=(
            "1. Generate common questions about the risks and potential complications after surgery, informed by the surgery details {surgery_details} and conversation {surgery_conversation}.\n"
            "2. Provide clear and honest answers outlining possible risks, their likelihood, and management strategies.\n"
            "3. Ensure transparency to help patients make informed decisions."
        ),
        expected_output=(
            "A set of well-formulated questions and answers concerning post-surgery risks and complications."
        ),
        agent=risks_complications_agent
    )

    medications_task = Task(
        description=(
            "1. Generate common questions about medications used during and after surgical procedures, based on the surgery details {surgery_details} and conversation {surgery_conversation}.\n"
            "2. Provide detailed answers covering types of medications, dosages, administration times, and potential side effects.\n"
            "3. Ensure the information is accurate and easily understandable for patients."
        ),
        expected_output=(
            "A set of well-formulated questions and answers related to post-surgery medications."
        ),
        agent=medications_agent
    )

    recovery_process_task = Task(
        description=(
            "1. Generate common questions about the recovery and rehabilitation process following surgery, utilizing the surgery details {surgery_details} and conversation {surgery_conversation}.\n"
            "2. Provide comprehensive answers addressing timelines, physical therapy, lifestyle adjustments, and strategies for optimal healing.\n"
            "3. Ensure the information assists patients in planning their recovery effectively."
        ),
        expected_output=(
            "A set of well-formulated questions and answers regarding the recovery process."
        ),
        agent=recovery_process_agent
    )

    patient_condition_task = Task(
        description=(
            "1. Generate common questions about assessing and documenting patient condition after surgery, informed by the surgery details {surgery_details} and conversation {surgery_conversation}.\n"
            "2. Provide detailed answers related to vital signs, recovery progress, and immediate post-operative observations.\n"
            "3. Ensure the information is clear and informative for patients and caregivers."
        ),
        expected_output=(
            "A set of well-formulated questions and answers concerning patient condition post-surgery."
        ),
        agent=patient_condition_agent
    )

    general_information_task = Task(
        description=(
            "1. Generate general questions about post-surgery practices, protocols, and advancements, leveraging the surgery details {surgery_details} and conversation {surgery_conversation}.\n"
            "2. Provide informative answers that cover best practices, safety measures, and innovations in post-operative care.\n"
            "3. Ensure the information is up-to-date and relevant to current post-surgery standards."
        ),
        expected_output=(
            "A set of well-formulated general post-surgery-related questions and answers."
        ),
        agent=general_information_agent
    )

    final_faq_compiler_task = Task(
        description=(
            "1. Gather the FAQs from the Postoperative Care Specialist, Risks and Complications Analyst, "
            "Medications Specialist, Recovery Process Advisor, Patient Condition Analyst, and General Post-Surgery Information Expert.\n"
            "2. Integrate these FAQs into a single, cohesive document.\n"
            "3. Ensure that the FAQs are organized by category and formatted consistently.\n"
            "4. Review the compiled FAQs for accuracy, clarity, and comprehensiveness.\n"
            "5. Finalize the FAQ document for distribution or publication."
        ),
        expected_output=(
            "A finalized, well-structured FAQ document encompassing all necessary post-surgery-related questions and answers."
        ),
        agent=final_faq_compiler_agent
    )

    # Defining the Crew

    faq_crew = Crew(
        agents=[
            faq_manager_agent,
            postoperative_care_agent,
            risks_complications_agent,
            medications_agent,
            recovery_process_agent,
            patient_condition_agent,
            general_information_agent,
            final_faq_compiler_agent
        ],
        tasks=[
            faq_manager_task,
            postoperative_care_task,
            risks_complications_task,
            medications_task,
            recovery_process_task,
            patient_condition_task,
            general_information_task,
            final_faq_compiler_task
        ],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential  # Ensures tasks are executed in order
    )

    # Initializing the Crew with necessary inputs

    initial_inputs = {
        'surgery_details': surgery_details,             # Detailed description of how the surgery was performed
        'surgery_conversation': surgery_conversation    # Transcript or notes of surgeon's conversation during surgery
    }

    # Executing the Crew

    result = faq_crew.kickoff(initial_inputs)
    return result
