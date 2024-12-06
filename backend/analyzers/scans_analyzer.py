import os
from dotenv import load_dotenv
from helper_functions.encode_image import encode_image

from groq import Groq


load_dotenv()

client = Groq()

def scans_analyzer(image_path):
    base64_image = encode_image(image_path)
    prompt = """
        Analyze the provided medical scan image (X-ray, MRI, or CT scan) and offer insights based on the given images to enhance the likelihood of surgical success.  Restrict your insights to information derived solely from the scan.

        **Input:**


        **Output Structure:**

        Provide insights in a structured format, using a numbered list. Each insight should clearly state the observation from the scan and its relevance to the surgery.  Use specific anatomical terminology where appropriate.

        **Example:**

        **Scan Image:**  (Example: An MRI showing a slightly enlarged gallbladder with thickened walls)


        **Output:**

        1. **Observation:** Gallbladder appears slightly enlarged with thickened walls.
        2. **Implication:**  Thickened gallbladder walls may indicate chronic inflammation or cholelithiasis.  Expect increased tissue density and potential for more challenging dissection. Consider using a smaller incision or more precise dissection techniques to avoid potential damage to adjacent structures. 
        3. **Observation:** No evidence of biliary ductal dilation or stones.
        4. **Implication:** The absence of biliary duct dilation suggests no common bile duct involvement, reducing the risk of injury to the common bile duct and the need for cholangiography.

        **Insights:**
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