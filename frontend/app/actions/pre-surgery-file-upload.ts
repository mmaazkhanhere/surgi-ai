import axios from "axios";

export async function uploadPreSurgeryFiles(
  files: any,
  surgery: string,
  patientHistory: string
) {
  try {
    const formData = new FormData();
    formData.append("prescription", files.prescription);
    formData.append("scan", files.scan);
    formData.append("lab_report", files.labReport);
    formData.append("operation", surgery); // Add operation
    formData.append("patient_history", patientHistory); // Add patient history

    const response = await axios.post(
      "http://localhost:8000/pre-surgery/upload",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    if (response.status === 200) {
      return { status: 200, data: response.data };
    } else {
      return { status: 400, message: "Error" };
    }
  } catch (error) {
    console.error(error);
    throw new Error("File upload failed");
  }
}
