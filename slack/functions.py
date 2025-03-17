import os
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables (optional)
load_dotenv(find_dotenv())

# RunPod API Configuration
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_ACCESS_TOKEN = os.environ["RUNPOD_ACCESS_TOKEN"]
CUSTOM_MODEL_NAME = "deepseek-r1"  # Replace with your fine-tuned DeepSeek-R1 model name

RUNPOD_API_URL = f"https://api.runpod.io/v2/{RUNPOD_ENDPOINT_ID}/run"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

class CustomDeepSeekR1(ChatOpenAI):
    """ Custom LangChain wrapper for fine-tuned DeepSeek-R1 model on RunPod """

    def _call(self, prompt, stop=None):
        payload = {
            "input": {
                "model": CUSTOM_MODEL_NAME,
                "prompt": prompt
            }
        }
        response = requests.post(RUNPOD_API_URL, json=payload, headers=HEADERS)

        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("output", "No response from model.")
        else:
            return f"Error: {response.text}"

# Function to draft an email using DeepSeek-R1 fine-tuned model
def draft_email(user_input, name="There"):
    chat = CustomDeepSeekR1(model_name="deepseek-r1")

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

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response

# Example Usage
#if __name__ == "__main__":
#    email_text = "Thank you for your interest in our services. Could you provide more details on your requirements?"
#    print(draft_email(email_text, name="John"))
