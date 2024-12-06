import axios from "axios";

export async function preSurgeryMedicine(file: any) {
  try {
    const response = await axios.post(
      "http://localhost:8000/pre-surgery/medicine",
      file,
      {
        headers: {
          "Content-Type": file.type, // e.g., "image/jpeg"
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
