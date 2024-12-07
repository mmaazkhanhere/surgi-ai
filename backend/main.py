from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import io

from analyzers.prescription_analyzer import prescription_analyzer
from analyzers.scans_analyzer import scans_analyzer
from analyzers.lab_reports_analyzer import lab_reports_analyzer

from agents.surgery_agent.surgery_agent import surgical_agent
from agents.pre_surgery_agent.surgical_procedure_agent import pre_surgical_report_agent

from helper_functions.to_markdown import to_markdown

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

@app.post("/pre-surgery/upload")
async def upload_files(
    prescription: UploadFile = File(...),
    scan: UploadFile = File(...),
    lab_report: UploadFile = File(...),
):
    try:
        # Function to validate and save images
        def process_file(file: UploadFile, file_name: str):
            file_content = file.file.read()
            try:
                image = Image.open(io.BytesIO(file_content))
                image.verify()  # Validate the image
                image = Image.open(io.BytesIO(file_content))
                image.load()  # Fully load to ensure it's decodable
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file_name} is not a valid image. Error: {e}",
                )

            # Save the image
            file_path = UPLOAD_DIR / file_name
            with file_path.open("wb") as f:
                f.write(file_content)

        # Process and save all files
        process_file(prescription, "prescription.jpg")
        process_file(scan, "scan.jpg")
        process_file(lab_report, "lab_report.jpg")

        prescription = prescription_analyzer('./uploads/prescription.jpg')
        print(f"Prescription: {prescription[:100]}")

        scan = scans_analyzer('./uploads/scan.jpg')
        print(f"Scan: {scan[:100]}")

        lab_report = lab_reports_analyzer('./uploads/lab_report.jpg')
        print(f"Lab Report: {lab_report[:100]}")

        state = {
            'prescription': prescription,
            'scan': scan,
            'lab_report': lab_report,
            'surgery': operation,
            'patient_history': patient_history,
            'prescription_report_analyzer_node': '',
            'lab_report_analyzer_node': '',
            'scan_report_analyzer_node': '',
            'instrumentation_report': '',
            'risk_analyzer_report': '',
            'anesthesia_consultant_report': '',
            'surgical_workflow_report': '',
            'emergency_protocol_advicer': '',
            'accumulator': ''
        }

        response = pre_surgical_report_agent(state)
        markdown_response = to_markdown(response)
        return markdown_response

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )