import axios from "axios";

export async function uploadPreSurgeryFiles(files: any) {
  try {
    const formData = new FormData();
    formData.append("prescription", files.prescription);
    formData.append("scan", files.scan);
    formData.append("lab_report", files.labReport);

    const response = await axios.post(
      "http://localhost:8000/pre-surgery/upload",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error(error);
    throw new Error("File upload failed");
  }
}
