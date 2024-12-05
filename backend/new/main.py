from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

from agent import surgical_agent

class DuringSurgery(BaseModel):
    surgeon_query: str
    patient_history: str

app: FastAPI = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def root():
    return {"message": "Welcome to SurgiAI"}


@app.post('/surgery')
async def surgical_query(input: DuringSurgery):
    state = {
        'surgeon_query': input.surgeon_query,
        'patient_history': input.patient_history,
        'conversation': [],
        'messages': [],
        'insight_accumulator_response': '',
        'max_iteration': 2,
        'current_iteration': 0
    }
    response = surgical_agent(state)
    return response
