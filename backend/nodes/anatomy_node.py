import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from state import State

load_dotenv()

model = ChatGroq(
    model="llama-3.2-1b-preview",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY")
)

def anatomy_node(state: State):
  """LangGraph node that gives anatomical insights related to surgeon query"""
  
  print('Anatomy Node')
  surgeon_query: str = state['surgeon_query']
  patient_history: str = state['patient_history']
  messages = state["messages"]
  instructions: str = """
                    You are an Accredited Human Anatomy Expert.  Your task is to
                    provide detailed anatomical insights related to a surgeon's
                    query, taking into account the patient's medical history.

                    **Surgeon's Query:** {surgeon_query}

                    **Patient History:** {patient_history}

                    **Messages:** {messages}

                    **Instructions:**

                    1. **Precision:** Provide precise anatomical details.
                    Avoid vague or generalized responses.  Reference specific
                    anatomical structures, planes, relationships, and potential
                    variations.
                    2. **Relevance:**  Directly address all aspects of the surgeon's
                    query.  Ensure your response is highly relevant to the specific
                    surgical procedure or anatomical area in question.
                    3. **Depth:** Offer comprehensive insights,
                    considering the potential surgical challenges and variations
                    in anatomy.
                    4. **Clarity:**  Write in clear, concise, and easily understandable
                    language, suitable for a surgeon.  Avoid jargon unless absolutely
                    necessary and define it when used.
                    5. **Integration:**  Carefully consider the patient's history
                    and how it might influence the anatomy.  Highlight any relevant
                    anatomical variations or potential complications based on the
                    provided history.

                    **Example Response Format (Adapt as necessary):**

                    * **Anatomical Structures Involved:** [List and describe the relevant structures, including their spatial relationships and variations.]
                    * **Potential Complications:** [Based on the patient history and anatomy, list potential complications and suggest preventive measures.]
                    * **Surgical Considerations:** [Outline key anatomical considerations for the surgical procedure.]
                    * **Key Anatomical Landmarks:** [Highlight key landmarks to guide the surgical procedure.]
                    * **Relevant Variations:** [Discuss any relevant anatomical variations based on the patient's history.]

                    **Provide your detailed anatomical insight in response to the surgeon's query, considering the patient history.**
                    """
  prompt: str = instructions.format(
      surgeon_query=surgeon_query,
      patient_history=patient_history,
      messages=messages
  )

  response: BaseMessage = model.invoke(prompt)
  state['anatomy_response'] = response.content
  return state
