"""
Chain creation utilities for NotesAssistant.
Links prompts, retrievers, and LLMs for answering questions.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from tkhtmlview import HTMLLabel
import tkinter as tk


def chainer(vectorstore, llm):
    """
    Links the prompt, retriever, and LLM to create a retrieval chain.

    Args:
        vectorstore: The vector store for document retrieval.
        llm: The language model instance.

    Returns:
        RetrievalChain: The constructed retrieval chain.
    """
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question at the end. 
    Answer in neat paragraphs with HTML formatting. Do not use backticks in your response.
    If the answer is not in the context, just say "I don't know"— do not make anything up.
    If the answer is in the context and the user specifies the statement "AOFS" which stands for "Answer only from slides" do not provide any other information other than what is in the context.
    In the event user does not specify that statement and If the answer is in the context, Explain the context using your own knowledge of that subject, it doesn't have to be in the context.
    Also include a fun explanation to the concept using different analogies.
    Finally, include a very brief intuitive explanation which helps me get the basic intution of what's going on.
    Context:
    {context}

    Question:
    {input}

    Answer:""")

    document_chain = create_stuff_documents_chain(llm, prompt)    
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def show_response_html_in_window(html_text):
    window = tk.Tk()
    window.title("NotesAssistant Response")
    window.geometry("800x600")

    html_label = HTMLLabel(window, html=html_text, background="white")
    html_label.pack(expand=True, fill="both", padx=10, pady=10)

    window.mainloop()

def print_response(retrieval_chain, prompt):
    """
    Invokes the retrieval chain with a prompt and displays the answer in a new window.

    Args:
        retrieval_chain: The retrieval chain to use.
        prompt (str): The question or prompt to send.

    Returns:
        None
    """
    response = retrieval_chain.invoke({"input":prompt})
    show_response_html_in_window(response["answer"])