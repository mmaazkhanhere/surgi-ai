import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import State

load_dotenv()

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    verbose=True,
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

def rewrite(state: State):
  print("---TRANSFORM QUERY---")
  messages = state["messages"]
  question = messages[0].content
  current_iteration = state.get('current_iteration', 0)
  msg = [
      HumanMessage(
          content=f""" \n
  Look at the input and try to reason about the underlying semantic intent / meaning. \n
  Here is the initial question:
  \n ------- \n
  {question}
  \n ------- \n
  Formulate an improved question: """,
      )
  ]
  response = model.invoke(msg)
  current_iteration += 1
  messages.append(response)
  return state
