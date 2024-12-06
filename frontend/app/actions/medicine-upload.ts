import axios from "axios";

export async function preSurgeryMedicine(file: File) {
  try {
    const formData = new FormData();
    formData.append("file", file); // 'file' should match FastAPI parameter

    const response = await axios.post(
      "http://localhost:8000/pre-surgery/medicine",
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
      return { status: 200, messages: "File uploaded successfully" };
    } else {
      return {
        status: response.status,
        message: response.data?.message || "Error",
      };
    }
  } catch (error) {
    console.error(error);
    return { status: 500, message: "Something went wrong" };
  }
}
