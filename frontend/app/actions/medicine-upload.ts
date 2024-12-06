import axios from "axios";

export async function preSurgeryMedicine(formData: any) {
  try {
    const response = await axios.post(
      `http://localhost:8000/pre-surgery/medicine`,
      {
        body: formData,
      }
    );

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
