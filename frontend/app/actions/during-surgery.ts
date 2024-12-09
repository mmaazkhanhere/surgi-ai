import axios from "axios";

export async function duringSurgery(
  surgeon_query: string,
  patient_history: string
) {
  try {
    const response = await axios.post(`https://surgi-ai.onrender.com/surgery`, {
      surgeon_query,
      patient_history,
    });

    console.log(response);
    if (response.status === 200) {
      return { status: 200, data: response.data };
    } else {
      return { status: 400, message: "Error" };
    }
  } catch (error) {
    console.log(error);
    return { status: 500, message: "Something went wrong" };
  }
}
