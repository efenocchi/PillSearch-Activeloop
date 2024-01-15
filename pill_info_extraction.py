from pathlib import Path
from typing import Dict, List, Union
from openai import OpenAI
import os
import base64
import requests
import base64
import requests
import yaml
import json

PILLS_JSON_FILE_CLEANED = "pills_info_cleaned.json"

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt_ocr(image_path: str):
    base64_image = encode_image(image_path)
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Tell me which is the number impressed on the element, pay attention that sometimes there is a background with a recurring writing and I don't want to receive that one. Pass me the information in a json file. If you don't find a text return the sentence "not_found". All the letters and number must be put in the same field. The json format should be like this: {\"number\": \"<item_number>\"}, where <item_number> is the number you found or "not_found" if you didn't find anything.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    # print(response.json())
    try:
        msg = response.json()["choices"][0]["message"]["content"]
        pill_id = msg.split('"number":')[1].split("}")[0].strip().strip('"')
    except:
        pill_id = "not_found"
    return pill_id


def extract_pill_number(image_folder: Union[str, Path], pills_info: Dict) -> Dict:
    """
    Extract the pill number from the images.
    """
    pills_code_extracted = None
    for el in pills_info:
        image_path = os.path.join(image_folder, pills_info[el]["image_path"])
        pills_code_extracted = gpt_ocr(image_path)
        print(f"Extracted for {image_path} the code: {pills_code_extracted}")
        pills_info[el]["pill_text"] = pills_code_extracted
    print(f"Extracted the following codes: {pills_code_extracted}")
    return pills_info


# client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-4-vision-preview",
#   messages=[
#     {
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "Tell me which is the pill color, which is the number impressed on it and what is the shape of the pill?"},
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#           },
#         },
#       ],
#     }
#   ],
#   max_tokens=300,
# )

# print(response.choices[0])

if __name__ == "__main__":
    with open(PILLS_JSON_FILE_CLEANED, "r") as file:
        pills_info = yaml.safe_load(file)

    image_folder = "./images"
    pills_code_extracted = extract_pill_number(image_folder, pills_info)

    with open(PILLS_JSON_FILE_CLEANED, "w") as json_file:
        json.dump(pills_code_extracted, json_file)
