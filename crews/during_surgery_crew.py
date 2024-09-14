import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            verbose=True,
            temperature=0.5,
            api_key=os.getenv('GOOGLE_API_KEY')
        )
def during_surgery_crew(surgeon_query: str, patient_history: str)-> str:
    # defining agents of the crew
    manager_agent = Agent(llm=llm_model,
                        role="Surgery Manager",
                        goal="Efficiently manage the surgeon's queries and gather precise insights from specialized agents.",
                        backstory="This agent ensures smooth coordination between the surgeon and specialized agents, focusing on concise, direct responses.",
                        verbose=False,
                        allow_delegation=False
                    )

    anatomy_specialist_agent = Agent(llm=llm_model,
                                    role="Anatomy Specialist",
                                    goal="Provide brief, clear guidance on anatomical aspects related to surgery.",
                                    backstory="Expert in anatomy, this agent focuses on clear and concise anatomical advice for surgeries.",
                                    verbose=False,
                                    allow_delegation=False
                                    )

    infection_prevision_agent = Agent(llm=llm_model,
                                    role="Infection Control Specialist",
                                    goal="Provide quick insights on infection prevention during surgery.",
                                    backstory="Specialized in infection risks, this agent provides direct and focused infection-related advice.",
                                    verbose=False,
                                    allow_delegation=False
                                    )

    risk_analysis_agent = Agent(llm=llm_model,
                                role="Risk Analyst",
                                goal="Assess risks related to the surgery and offer concise recommendations.",
                                backstory="Expert in surgical risks, providing concise evaluations of potential complications.",
                                verbose=False,
                                allow_delegation=False
                                )

    expert_surgeon_agent = Agent(llm=llm_model,
                                role="Expert Surgeon",
                                goal="Synthesize insights from specialized agents and provide a concise response to the surgeon.",
                                backstory="This agent summarizes the findings from all specialists and delivers a clear, concise response to the surgeon.",
                                verbose=False,
                                allow_delegation=True
                                )

    # Defining tasks for the agents
    manager_task = Task(
        description=("Manage the surgeon's query  '{surgeon_query}' and assign it to the appropriate agents."),
        expected_output="Ensure the query is addressed by the appropriate specialized agents.",
        agent=manager_agent
    )

    anatomy_task = Task(
        description=("Provide a direct answer to anatomy-related queries in the context of the surgery."),
        expected_output="Give a concise explanation relevant to the anatomy aspect of the surgeon's query {surgeon_query} respective to patient history {patient_history}.",
        agent=anatomy_specialist_agent
    )

    infection_prevision_task = Task(
        description=("Offer concise advice on preventing infection during the surgery."),
        expected_output="Provide direct guidance on infection prevention tailored to the patient's history {patient_history} associated with surgeon query {surgeon_query}.",
        agent=infection_prevision_agent
    )

    risk_analysis_task = Task(
        description=("Evaluate the risk associated with the surgeon's query {surgeon_query} and provide a focused response keeping the patient history {patient_history} in context."),
        expected_output="Offer a brief risk analysis based on the surgical scenario and patient history.",
        agent=risk_analysis_agent
    )

    expert_surgeon_task = Task(
        description=("Synthesize input from all agents and provide a concise final answer to the surgeon query {surgeon_query} respective to patient history {patient_history}."),
        expected_output="Deliver a direct, concise response to the surgeon’s query {surgeon_query}",
        async_execution=True,
        agent=expert_surgeon_agent
    )

    # Defining the crew
    surgical_crew = Crew(
        agents=[manager_agent, anatomy_specialist_agent, infection_prevision_agent, risk_analysis_agent, expert_surgeon_agent],
        tasks=[manager_task, anatomy_task, infection_prevision_task, risk_analysis_task, expert_surgeon_task],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential
    )

    result = surgical_crew.kickoff({'surgeon_query': surgeon_query, 'patient_history': patient_history})
    return result