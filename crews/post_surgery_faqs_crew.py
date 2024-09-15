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

def surgery_post_faq_crew(surgery_details: str, surgery_conversation: str) -> str:
    """
    Crew of agents responsible for generating comprehensive post-surgery FAQs.
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
        allow_delegation=True
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

    # creating tasks for the FAQ agents

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
            "2. Integrate atleast twenty FAQs into a single, cohesive document.\n"
            "3. Ensure that the FAQs are organized by category and formatted consistently.\n"
            "4. Review the compiled FAQs for accuracy, clarity, and comprehensiveness.\n"
            "5. Finalize the FAQ document for distribution or publication."
        ),
        expected_output=(
            "A finalized, well-structured FAQ document encompassing all necessary post-surgery-related questions and answers."
        ),
        agent=final_faq_compiler_agent
    )

    # creating crew

    faq_crew = Crew(
        agents=[
            faq_manager_agent,
            postoperative_care_agent,
            recovery_process_agent,
            general_information_agent,
            final_faq_compiler_agent
        ],
        tasks=[
            faq_manager_task,
            postoperative_care_task,
            recovery_process_task,
            general_information_task,
            final_faq_compiler_task
        ],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential  
    )
    # Initializing the Crew with necessary inputs

    initial_inputs = {
        'surgery_details': surgery_details,             
        'surgery_conversation': surgery_conversation   
    }

    # Executing the Crew

    result = faq_crew.kickoff(initial_inputs)
    return result
