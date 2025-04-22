# LangGraph + LCEL version of LLM + tool chaining
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langgraph.graph import END, StateGraph
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema.document import Document
import re, os, json
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, ValidationError, field_validator
from typing import TypedDict, Optional, List
from fpdf import FPDF

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']

# --- FAISS vector store setup ---
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
initial_docs = [Document(page_content=f"Rule {i}") for i in range(1, 4)]
vectorstore = FAISS.from_documents(initial_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# --- Tools ---
def getdiscount(prodname: str, totalamount: int) -> int:
    print(f"🔧 getdiscount called with prodname={prodname}, totalamount={totalamount}")
    return 5 if prodname == "AWS" and totalamount >= 10000 else 2

discount_tool = Tool(
    name="get_discount",
    func=lambda input: str(getdiscount(**json.loads(input))),
    description="Get discount based on product name and total amount. Input should be a JSON string with 'prodname' and 'totalamount'."
)

# --- LLM Output Schema ---
class LLMResponse(BaseModel):
    action: Optional[str] = None
    customer_name: Optional[str] = None
    product_name: Optional[str] = None
    amount: Optional[int] = None
    month: Optional[str] = None
    contract_length: Optional[str] = None

# --- Graph State ---
class GraphState(TypedDict):
    question: str
    parsed: Optional[LLMResponse]
    result: Optional[str]
    response: Optional[str]
    customer_name: Optional[str]
    product_name: Optional[str]
    amount: Optional[int]
    month: Optional[str]
    contract_length: Optional[str]
    current_step: Optional[str]
    action: Optional[str]
    awaiting_field: Optional[str]
    generate_pdf: Optional[str]
    pdf_generated: Optional[bool]

# --- LLM Setup ---
llm = ChatOpenAI(temperature=0, api_key=api_key, model="gpt-4o")
memory = ConversationBufferMemory(return_messages=True)

# --- LangGraph Nodes ---
def entry_node(state: GraphState):
    #print("🔹 entry_node triggered")

    if state.get("pdf_generated"):
        return {
            **state,
            "response": "PDF has already been generated. Let me know if you need anything else!",
            "question": "",
            "awaiting_field": None
        }

    field = state.get("awaiting_field")
    user_input = state.get("question", "")

    if field and user_input:
        #print(f"✍️ Saving '{user_input}' to field: {field}")
        if field == "amount":
            try:
                user_input = int(user_input)
            except ValueError:
                print("⚠️ Warning: amount could not be converted to integer. Keeping as string.")

        # ✅ Clear awaiting_field if generate_pdf response is provided
        next_state = {
            **state,
            field: user_input,
            "question": "",
            "parsed": None,
        }
        if field == "generate_pdf":
            next_state["question"] = ""  # prevent re-parsing blank question
        return next_state

    return {**state, "response": "Hi! How can I help you today?"}


def parse_request_node(state: GraphState):
    #print("🔹 parse_request_node triggered")
    question = state.get("question", "")
    #print("state.question is =", question)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
    '''You are a helpful assistant that analyzes customer requests to generate price quotations.
Always respond in valid JSON with ALL of the following keys:
- action (string: e.g. "generate_quote")
- customer_name (string or null)
- product_name (string or null)
- amount (integer or null)
- month (string or null)
- contract_length (string or null)

Respond ONLY with raw JSON. No markdown or explanations.

Example:
{{"action": "generate_quote", "customer_name": "XYZ", "product_name": "AWS", "amount": 15000, "month": "April", "contract_length": "12 months"}}'''
),
        HumanMessagePromptTemplate.from_template("User input: {question}")
    ])

    try:
        messages = prompt.format_messages(question=question)
    except KeyError as e:
        print("🛑 Prompt formatting failed due to missing key:", e)
        return {**state, "response": f"Template format error: missing key {e}"}

    raw = llm.invoke(messages)
    #print("LLM Raw Response:", raw)

    try:
        content = raw.content.strip()
        #print("🧹 Raw content before cleanup:", content)

        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        parsed_json = json.loads(content)
        #print("📦 Parsed JSON:", parsed_json)

        for key in ["action", "customer_name", "product_name", "amount", "month", "contract_length"]:
            parsed_json.setdefault(key, None)

        parsed = LLMResponse(**parsed_json)
        #print("✅ Parsed LLM response:", parsed)
        return {
            **state,
            "parsed": parsed,
            "action": parsed.action or state.get("action") or "unknown_action",
            "customer_name": parsed.customer_name or state.get("customer_name"),
            "product_name": parsed.product_name or state.get("product_name"),
            "amount": parsed.amount or state.get("amount"),
            "month": parsed.month or state.get("month"),
            "contract_length": parsed.contract_length or state.get("contract_length"),
            "awaiting_field": None
        }
    except Exception as e:
        print(f"❌ LLM parsing failed: {e}")
        return {**state, "response": "Sorry, I didn't understand that. Could you rephrase?", "action": "unknown_action"}


def ask_customer_name_node(state: GraphState):
    #print("🔹 ask_customer_name_node triggered")
    if not state.get("customer_name"):
        return {**state, "response": "What is the customer name?", "awaiting_field": "customer_name"}
    return {**state, "awaiting_field": None}

def collect_quote_details_node(state: GraphState):
    #print("🔹 collect_quote_details_node triggered")
    product = state.get("product_name")
    amount = state.get("amount")
    month = state.get("month")
    contract = state.get("contract_length")

    if not state.get("customer_name"):
        return {**state, "response": "What is the customer name?", "awaiting_field": "customer_name"}
    if not product:
        return {**state, "response": "What is the product name?", "awaiting_field": "product_name"}
    if not amount:
        return {**state, "response": "What is the total amount?", "awaiting_field": "amount"}
    if not month:
        return {**state, "response": "What is the purchase month?", "awaiting_field": "month"}
    if not contract:
        return {**state, "response": "What is the contract length?", "awaiting_field": "contract_length"}

    return {**state, "awaiting_field": None}


def finalize_quote_node(state: GraphState):
    #print("🔹 finalize_quote_node triggered")

    if state.get("pdf_generated"):
        print("🔁 Skipping quote regeneration because PDF is already generated.")
        return state

    product = state.get("product_name")
    amount = state.get("amount")
    customer = state.get("customer_name")
    month = state.get("month")
    contract = state.get("contract_length")

    if product and amount and customer and month and contract:
        discount = getdiscount(product, amount)
        discounted_price = amount * (1 - discount / 100)
        quote = (
            f"Final quote for {customer}:\n"
            f"- Product: {product}\n"
            f"- Amount: ${amount}\n"
            f"- Discount Applied: {discount}%\n"
            f"- Discounted Total: ${discounted_price:.2f}\n"
            f"- Purchase Month: {month}\n"
            f"- Contract Length: {contract}"
        )

        # ✅ Ask about PDF if not already decided
        if not state.get("generate_pdf") and not state.get("pdf_generated"):
            return {
                **state,
                "result": quote,
                "response": quote + "\n\nWould you like to generate a PDF quote?",
                "awaiting_field": "generate_pdf"
            }

        return {
            **state,
            "result": quote,
            "response": quote,
            "awaiting_field": None
        }

    return {**state, "response": "Missing information for quote."}


def output_node(state: GraphState):
    #print("🔹 output_node triggered")
    if state.get("awaiting_field") == "generate_pdf" and not state.get("pdf_generated"):
        answer = state.get("question", "").lower().strip()
        if answer in ["yes", "y"]:
            print("📄 Generating PDF...")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in state.get("result", "").split("\n"):
                pdf.cell(200, 10, txt=line, ln=True)
            filename = f"quote_{state.get('customer_name', 'customer')}.pdf"
            pdf.output(filename)
            return {
                **state,
                "response": f"PDF generated: {filename}",
                "awaiting_field": None,
                "question": "",
                "pdf_generated": True,
                "generate_pdf": None,
                "action": "pdf_complete"
            }

    if state.get("pdf_generated"):
        return {**state, "response": "Let me know if you need anything else!"}
    
    return {
        **state,
        "response": state.get("response") or state.get("result") or "No response."
    }


def generate_pdf_node(state: GraphState):
    print("📄 generate_pdf_node triggered")
    if state.get("generate_pdf", "").lower() in ["yes", "y"] and not state.get("pdf_generated"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in state.get("result", "").split("\n"):
            pdf.cell(200, 10, txt=line, ln=True)
        filename = f"quote_{state.get('customer_name', 'customer')}.pdf"
        pdf.output(filename)
        return {
            # Reset state after PDF generation
            "question": "",
            "parsed": None,
            "result": None,
            "response": f"PDF generated: {filename}\n\nLet me know if you need anything else!",
            "customer_name": None,
            "product_name": None,
            "amount": None,
            "month": None,
            "contract_length": None,
            "current_step": None,
            "action": None,
            "awaiting_field": None,
            "generate_pdf": None,
            "pdf_generated": None
        }

    return {**state, "response": "Let me know if you need anything else!"}



# --- Graph ---
graph = StateGraph(GraphState)
graph.add_node("entry", entry_node)
graph.add_node("parse_request", parse_request_node)
graph.add_node("ask_customer", ask_customer_name_node)
graph.add_node("collect_quote_details", collect_quote_details_node)
graph.add_node("finalize_quote", finalize_quote_node)
graph.add_node("output", output_node)
graph.add_node("generate_pdf_node", generate_pdf_node)

graph.set_entry_point("entry")


graph.add_conditional_edges("entry", lambda s: (
    "collect_quote_details" if s.get("awaiting_field") else "parse_request"
    if s.get("question") else "output"
))


graph.add_conditional_edges("parse_request", lambda s: (
    "collect_quote_details" if s.get("action") == "generate_quote" and s.get("customer_name") else "ask_customer" if s.get("action") == "generate_quote" else "output"
))
graph.add_edge("ask_customer", "collect_quote_details")
graph.add_conditional_edges("collect_quote_details", lambda s:
    "output" if s.get("awaiting_field") else "finalize_quote"
)
graph.add_edge("finalize_quote", "output")

graph.add_conditional_edges("output", lambda s: (
    "generate_pdf_node"
    if isinstance(s.get("generate_pdf"), str) and s.get("generate_pdf").lower() in ["yes", "y"] and not s.get("pdf_generated")
    else END
))

graph.add_edge("generate_pdf_node", END)
graph.set_finish_point("generate_pdf_node" if "generate_pdf_node" in graph.nodes else "output")


compiled_graph = graph.compile()

# --- Run Conversation Loop ---
print("\n🤖 Agent: Hi! How can I help you today?")
state = {"question": "Hi"}
while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    field = state.get("awaiting_field")
    if field:
        state = {
            **state,
            "question": user_input
        }
    else:
        state = {
            **state,
            "question": user_input,
            "parsed": None,
            "action": None
        }

    output = compiled_graph.invoke(state)
    print("🤖 Agent:", output["response"])
    state.update(output)
