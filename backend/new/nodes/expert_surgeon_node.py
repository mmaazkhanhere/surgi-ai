import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage

from tools import query_database
from state import State

load_dotenv()

@tool
def query_database(surgeon_query: str):
  """Query the pinecone database for the surgeon query"""
  print("Query database")
  pinecone_api_key = os.getenv('PINECONE_API_KEY')
  pc = Pinecone(api_key=pinecone_api_key)
  index_name = "surgical-assistant"
  index = pc.Index(index_name)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
  vector_store = PineconeVectorStore(index=index, embedding=embeddings)
  retriever = vector_store.as_retriever()
  retrieved_information = retriever.invoke(surgeon_query)
  return retrieved_information

tools = [query_database]

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    verbose=True,
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)
model_with_tools = model.bind_tools(tools)

def expert_surgeon_node(state: State):
  """ LangGraph node that give surgical insights related to surgeon query"""

  print("Expert Surgeon")
  surgeon_query: str = state['surgeon_query']
  insight_accumulator_response: str = state['insight_accumulator_response']
  patient_history: str = state['patient_history']

  instructions = """
    You are a Certified Surgical Consultant providing concise, high-quality
    answers to surgeon queries.
    Use your extensive knowledge of surgical procedures, anatomy, infection
    prevention, and potential
    complications to formulate responses.  Prioritize clarity and accuracy;
    avoid unnecessary detail.

    **Input:**

    * Surgeon's Query: {surgeon_query}


    **Instructions:**

    1. Analyze the surgeon's query in the context of the provided patient history
    {patient_history}
    and insights.
    2. Synthesize the information into a clear and actionable response, directly
    answering the question.
    3. Prioritize conciseness.  Assume the surgeon is under time pressure and
    needs brief, critical information.
    4. Your response must be conversational in nature, suitable for a direct
    dialogue with a surgeon.
    5. If insufficient information is provided to answer definitively,
    acknowledge this and suggest further investigations or data needed.

  """
  prompt: str = instructions.format(
    surgeon_query=surgeon_query,
    patient_history=patient_history
  )

  state['messages'].append(HumanMessage(content=prompt))
  response: BaseMessage = model_with_tools.invoke(state['messages'])
  state['messages'] = [response]
  state['expert_surgeon_response'] = response.content
  print(state['expert_surgeon_response'])
  return state


