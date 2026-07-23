import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

endpoint = os.getenv("endpoint")
deployment_name = os.getenv("deployment_name")
api_key = os.getenv("api_key")

print("endpoint:", endpoint)
print("deployment_name:", deployment_name)

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)