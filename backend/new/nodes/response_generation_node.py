import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage

from tools import query_database
from state import State

load_dotenv()

tools = [query_database]

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    verbose=True,
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

def response_generation_node(state: State):
  """LangGraph node that generates final response to surgeon query"""

  print("Response Generation")
  surgeon_query = state['surgeon_query']
  insights = state['insight_accumulator_response']
  instructions = """ 
          Based on insights generated, {insights}, and surgeon query, give a response
          to surgeon query {surgeon_query}

          Respond directly to the surgeon's question, as if you were speaking directly to them.
          Do not include explanations or justifications for how you reached your response.
          The response should be a single short paragraph containing. The response must be
          less than 2 sentences and 50 words


          **Example:**

          Surgeon's Query: What is the risk of infection post-appendectomy in a diabetic patient?

          Response:  The risk of infection is moderately elevated in diabetic patients due to
          impaired wound healing.   Prophylactic antibiotics are recommended, and close monitoring for
          signs of infection in the first 72 hours is crucial.

          **Response:**
        """
  prompt: str = instructions.format(
    surgeon_query=surgeon_query,
    insights=insights
  )
  response: BaseMessage = model.invoke(prompt)
  state['conversation'].append(response.content)
  state['answer'] = response.content
  return state