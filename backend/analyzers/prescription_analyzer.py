import os
from dotenv import load_dotenv
from helper_functions.encode_image import encode_image

from groq import Groq


load_dotenv()

client = Groq()

def prescription_analyzer(image_path):
    base64_image = encode_image(image_path)
    prompt = """Analyze the provided prescription image and extract relevant 
        insights to improve the success probability of the surgical procedure.  Focus exclusively on information present within the prescription itself. Do not hallucinate information.

        **Output Structure:**

        * **medication:** List of all medications prescribed, including dosage and frequency.
        * **allergies:** List of any documented allergies.
        * **relevant_conditions:** List of relevant medical conditions impacting the surgery.
        * **contraindications:** List of contraindications for specific surgical procedures or medications.
        * **preoperative_instructions:** Any specific preoperative instructions mentioned.
        * **insights:** A summary of how these prescription details can affect the surgical plan, including potential risks and precautions to take during the surgery based on the prescription.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )

    return chat_completion.choices[0].message.content