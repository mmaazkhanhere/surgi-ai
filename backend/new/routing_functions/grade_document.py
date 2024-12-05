import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from state import State, Grade

load_dotenv()

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    verbose=True,
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

structure_model = model.with_structured_output(Grade)

def grade_document(state: State):
  print("---CHECK RELEVANCE---")
  # Prompt
  prompt = PromptTemplate(
      template="""You are a grader assessing relevance of a retrieved document 
      to a user question. \n
      Here is the retrieved document: \n\n {context} \n\n
      Here is the user question: {question} \n
      If the document contains keyword(s) or semantic meaning related to the 
      user question, grade it as relevant. \n
      Give a binary score 'yes' or 'no' score to indicate whether the document 
      is relevant to the question.""",
      input_variables=["context", "question"],
  )
  # Chain
  chain = prompt | structure_model
  messages = state["messages"]
  last_message = messages[-1]
  max_iteration = state.get("max_iteration", 1)
  current_iteration = state.get('current_iteration', 0)

  question = messages[0].content
  docs = last_message.content
  scored_result = chain.invoke({"question": question, "context": docs})
  score = scored_result.grade
  if score == "no" and current_iteration < max_iteration:
      print("---DECISION: DOCS NOT RELEVANT---")
      print(score)
      return "Rewrite"
  else:
      print("---DECISION: DOCS RELEVANT---")
      print(score)
      return 'Insight Accumulator'

