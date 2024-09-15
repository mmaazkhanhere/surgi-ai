import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from helper_functions.pinecone_vector_store import pinecone_vector_store
from helper_functions.pinecone_vector_store import embeddings

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            verbose=True,
            temperature=0.5,
            api_key=os.getenv('GOOGLE_API_KEY')
        )

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


def during_surgery_crew(surgeon_query: str, patient_history: str)-> str:
    
    # defining agents of the crew
    manager_agent = Agent(llm=llm_model,
                        role="Certified Operating Room Manager",
                        goal="Effectively manage surgeon queries and consult specialized agents to gather real-time insights for informed decision-making.",
                        backstory="This agent is tasked with managing the flow of queries from the surgeon {surgeon_query} during surgery. You consult with specialized agents, using patient history {patient_history}",
                        verbose=True,
                        allow_delegation=False
                    )

    anatomy_specialist_agent = Agent(llm=llm_model,
                                    role="Accredited Human Anatomy Expert",
                                    goal="Provide clear, concise, and accurate anatomical insights in response to surgeon queries.",
                                    backstory="The agent specialize in human anatomy and are responsible for delivering precise anatomical details during surgery. When the surgeon asks a query {surgeon_query}, agent provide relevant and easily understandable insights that aid in the surgical process",
                                    verbose=True,
                                    allow_delegation=False,
                                    tools=[query_pinecone]
                                )

    infection_prevention_agent = Agent(llm=llm_model,
                                    role="Certified Infection Preventionist",
                                    goal="Provide concise and accurate infection prevention steps to minimize the risk of infections related to the surgeon's query.",
                                    backstory="The agent specialize in infection control and is responsible for advising on best practices to reduce infection risk during surgery. When the surgeon raises a query {surgeon_query}, the provide provide clear, brief, and actionable precautions to maintain a sterile environment and ensure patient safety.",
                                    verbose=True,
                                    allow_delegation=False
                                )

    risk_analysis_agent = Agent(llm=llm_model,
                                role="Licensed Surgical Risk Mitigation Expert",
                                goal="Provide insights on potential risks related to the surgeon's query, taking the patient's history into account.",
                                backstory="Agent specialize in surgical risk management and assess potential risks during procedures. When the surgeon asks a query {surgeon_query}, you evaluate the patient's history {patient_history} and provide concise, accurate insights to help mitigate risks and ensure patient safety.",
                                verbose=True,
                                allow_delegation=False
                            )

    expert_surgeon_agent = Agent(llm=llm_model,
                                role="Certified Surgical Consultant",
                                goal="Provide concise, well-rounded responses to surgeon queries by synthesizing insights from specialized agents and considering the patient's history",
                                backstory="This agent serve as the primary consultant during surgery, integrating insights from specialized agents to provide the surgeon with clear, actionable responses. The agent focus is on delivering accurate, to-the-point guidance that factors in both expert insights and the patient's medical history.",
                                verbose=True,
                                allow_delegation=True,
                                tools=[query_pinecone]
                            )

    # Defining tasks for the agents
    manager_task = Task(description=(
                            "1. Receive and understand the surgeon's query: {surgeon_query}.\n"
                            "2. Consult with the appropriate specialized agents, including the "
                                "Accredited Human Anatomy Expert, Certified Infection Preventionist, "
                                "and Licensed Surgical Risk Mitigation Expert.\n"
                            "3. Ensure all insights are accurately gathered and consider the patient's history: {patient_history}.\n"
                            "4. Compile the insights into a concise and actionable response "
                                "that helps the surgeon make informed decisions during surgery.\n"
                            "5. Monitor the situation continuously and provide updates if new risks or complications arise.\n"
                        ),
                        expected_output=(
                            "A well-organized response to the surgeon query that includes "
                            "insights from specialized agents, a summary of patient history, "
                            "and practical recommendations for next steps."
                        ),
                        agent=manager_agent,
    )


    anatomy_task = Task(description=(
                        "1. Receive the surgeon's query: {surgeon_query}.\n"
                        "2. Analyze the query to identify relevant anatomical structures, "
                            "functions, and relationships based on the specific area of concern and patient history {patient_history}.\n"
                        "3. Provide clear and concise anatomical insights, ensuring the response is easily understood "
                            "by the surgeon during the surgery.\n"
                        "4. Highlight any critical anatomical considerations, such as proximity to vital organs, "
                            "nerves, or blood vessels, that may impact the surgical procedure.\n"
                        "5. Continuously be available to provide further clarification or additional insights as required."
                    ),
                    expected_output=(
                        "A detailed, yet easy-to-understand, anatomical explanation that directly addresses "
                        "the surgeon's query and aids in making informed decisions during surgery."
                    ),
                    agent=anatomy_specialist_agent,
                    tools=[query_pinecone]
                )


    infection_prevention_task = Task(description=(
                                    "1. Receive the surgeon's query: {surgeon_query}.\n"
                                    "2. Analyze the query to assess infection risks based on the specific surgical context and patient history {patient_history}.\n"
                                    "3. Provide concise and actionable infection prevention steps, including recommended sterilization procedures, "
                                        "use of protective barriers, or handling of surgical tools.\n"
                                    "4. Highlight any critical factors that could increase infection risk, such as patient conditions or environmental concerns.\n"
                                    "5. Ensure the recommendations are practical, easy to implement during surgery, and aimed at maintaining a sterile environment."
                                ),
                                expected_output=(
                                    "A clear and actionable response that includes infection prevention strategies tailored "
                                    "to the surgeon's query and the current surgical situation, aimed at minimizing infection risk and ensuring patient safety."
                                ),
                                agent=infection_prevention_agent,
                            )


    risk_analysis_task = Task(description=(
                                "1. Receive the surgeon's query: {surgeon_query}.\n"
                                "2. Analyze the query in conjunction with the patient's history: {patient_history}.\n"
                                "3. Identify potential risks based on the patient's condition, surgical procedure, and any pre-existing factors.\n"
                                "4. Provide clear and concise risk mitigation strategies that are specific to the current surgery and patient.\n"
                                "5. Highlight any critical risk factors that could affect the outcome, offering actionable insights to reduce these risks.\n"
                                "6. Ensure the response is timely and easy for the surgeon to implement during the procedure."
                            ),
                            expected_output=(
                                "A detailed yet concise risk analysis that considers the surgeon's query and the patient's history, "
                                "offering practical risk mitigation strategies to ensure patient safety and minimize potential complications."
                            ),
                            agent=risk_analysis_agent,
                        )


    expert_surgeon_task = Task(description=(
                                "1. Receive the surgeon's query: {surgeon_query}.\n"
                                "2. Gather insights from specialized agents, including the Accredited Human Anatomy Expert, Certified Infection Preventionist, and Licensed Surgical Risk Mitigation Expert.\n"
                                "3. Synthesize the information provided by each agent, ensuring that the insights are cohesive and address all aspects of the surgeon's query.\n"
                                "4. Consider the patient's history: {patient_history} to ensure the response is tailored to the patient’s specific condition and medical background.\n"
                                "5. Provide a concise, clear, and actionable response that aids in making informed surgical decisions.\n"
                            ),
                            expected_output=(
                                "Deliver a direct, concise response to the surgeons query {surgeon_query} in 2 sentence. Dont give general answer"
                            ),
                            agent=expert_surgeon_agent,
                            tools=[query_pinecone]
                        )


    # Defining the crew
    surgical_crew = Crew(
        agents=[manager_agent, anatomy_specialist_agent, infection_prevention_agent, risk_analysis_agent, expert_surgeon_agent],
        tasks=[manager_task, anatomy_task, infection_prevention_task, risk_analysis_task, expert_surgeon_task],
        verbose=True,
        manager_llm=llm_model,
        process=Process.sequential
    )

    result = surgical_crew.kickoff({'surgeon_query': surgeon_query, 'patient_history': patient_history})
    return result