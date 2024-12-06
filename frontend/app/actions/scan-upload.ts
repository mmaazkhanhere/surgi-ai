import axios from "axios";

export async function preSurgeryScans(file: any) {
  try {
    const formData = new FormData();
    formData.append("file", file); // 'file' should match FastAPI parameter

    const response = await axios.post(
      "http://localhost:8000/pre-surgery/scans",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data", // Optional: Axios sets this automatically
        },
      }
    );

    console.log(response);
    if (response.status === 200) {
      return { status: 200, messages: "File uploaded successfully" };
    } else {
      return { status: 400, message: "Error" };
    }
  } catch (error) {
    console.log(error);
    return { status: 500, message: "Something went wrong" };
  }
}
