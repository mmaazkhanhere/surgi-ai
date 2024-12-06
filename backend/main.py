from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import io

from analyzers.prescription_analyzer import prescription_analyzer
from agents.surgery_agent.surgery_agent import surgical_agent

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

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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

@app.post("/pre-surgery/medicine")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file content
        file_content = await file.read()

        # Validate the file as an image
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()  # Check if it's a valid image
            image = Image.open(io.BytesIO(file_content))
            image.load()  # Reopen the image to ensure it's decodable
        except Exception as e:
            print(f"Image validation error: {e}")
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # Save the image as a file
        file_path = UPLOAD_DIR / "uploaded_medicine.jpg"
        with file_path.open("wb") as f:
            f.write(file_content)

        analysis = prescription_analyzer('./uploads/uploaded_medicine.jpg')
        print(analysis)

        return JSONResponse(
            content={"status": "success", "message": "File uploaded and saved."},
            status_code=200,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )
    