import os
import requests
import time
from langchain.chains import LLMChain
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation
from typing import List
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables (optional)
load_dotenv(find_dotenv())

# RunPod API Configuration
#RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
#RUNPOD_ACCESS_TOKEN = os.environ["RUNPOD_ACCESS_TOKEN"]
RUNPOD_ENDPOINT_ID = "94js71uw1j9ihj"
RUNPOD_ACCESS_TOKEN = "rpa_7R9GRZPL1KSKOWTS1LVBXI2VEKTR880AA2MVC93P179q63"
CUSTOM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Replace with your fine-tuned DeepSeek-R1 model name

RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"


def check_status(job_id):
    headers = {
        'Authorization': f'Bearer {RUNPOD_ACCESS_TOKEN}'
    }
    response = requests.get(f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}', headers=headers)
    return response


class CustomDeepSeekR1(LLM):
    """ Custom LangChain wrapper for fine-tuned DeepSeek-R1 model on RunPod """
    def _call(self, prompt: str, stop=None) -> str:
        #super()._call(**kwargs)
        HEADERS = {
            "Authorization": f"Bearer {RUNPOD_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": {
                "model": CUSTOM_MODEL_NAME,
                "prompt": prompt,
                "sampling_params": {
                    "max_tokens": 2048,
                    "temperature": 0
                }
            }
        }

        response = requests.post(RUNPOD_API_URL, json=payload, headers=HEADERS)
        job_id = response.json()["id"]

        try:
            while True:
                status = check_status(job_id)
                status_json = status.json()
                if status_json.get('status') == "COMPLETED":
                    response_data = status.json().get('output')[0].get('choices')[0].get('tokens')[0]
                    break
                time.sleep(4)
        except Exception as e:
            print(f"❌ Error fetching job status: {e}")   
        if '</think' in response_data:
            response_data = response_data.split('</think>')[1].strip()
        return response_data
    
    @property
    def _llm_type(self) -> str:
        return "custom-runpod-deepseek"



# Function to draft an email using DeepSeek-R1 fine-tuned model
def draft_email(user_input, name="There"):

    template = """
    You are a helpful assistant that drafts an email reply based on an a new email.
    Your goal is to help the user quickly create a perfect email reply.
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    Make sure to sign off with {signature}.
    """

    signature = f"Kind regards, \n"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chat = CustomDeepSeekR1(model_name=CUSTOM_MODEL_NAME)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)
    if '</think' in response:
        response = response.split('</think>')[1].strip()
    #print("final response = ", response)
    return response
