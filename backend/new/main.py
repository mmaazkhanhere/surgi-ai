from fastapi import FastAPI
from pydantic import BaseModel

class DuringSurgery(BaseModel):
    surgeon_query: str
    patient_history: str

app: FastAPI = FastAPI()

app.get('/')
async def root():
    return {"message": "SurgiAI"}


app.post('/surgery')
async def surgery(input: DuringSurgery):
    return {"message": "Surgery"}