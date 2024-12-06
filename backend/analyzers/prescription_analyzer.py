import os
from dotenv import load_dotenv
from helper_functions.encode_image import encode_image

from groq import Groq


load_dotenv()

client = Groq()

def prescription_analyzer(image_path):
    base64_image = encode_image(image_path)
    prompt = """Analyze the provided doctor's prescription image. Identify key medications, dosages, frequencies, and any relevant patient information (e.g., allergies, medical history).  Extract insights crucial for surgical procedure planning, such as potential drug interactions, contraindications, or necessary pre-operative adjustments.  Present these insights concisely."""

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