from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
def init_llm():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        convert_system_message_to_human=True,
        temperature=0.0,
    )

def init_embeddings():
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this sentence for answering questions:"
    )