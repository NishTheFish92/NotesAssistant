"""
Chain creation utilities for NotesAssistant.
Links prompts, retrievers, and LLMs for answering questions.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from settings import TOP_K

PROMPT = ChatPromptTemplate.from_template("""
    Use the following context to answer the question at the end.
    Answer in neatly formatted paragraphs and with Markdown formatting. Do not use backticks in your response.
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


class RetrievalChain:
    """
    Retrieve the top-k chunks for a question, stuff them into the prompt, ask the LLM.

    invoke() returns the same shape the old LangChain retrieval chain did:
    {"input": question, "context": [Document, ...], "answer": str}
    """

    def __init__(self, vectorstore, llm, k=TOP_K):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.answer_chain = PROMPT | llm | StrOutputParser()

    def invoke(self, inputs):
        question = inputs["input"]
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self.answer_chain.invoke({"context": context, "input": question})
        return {"input": question, "context": docs, "answer": answer}


def chainer(vectorstore, llm):
    """
    Links the prompt, retriever, and LLM to create a retrieval chain.

    Args:
        vectorstore: The vector store for document retrieval.
        llm: The language model instance.

    Returns:
        RetrievalChain: The constructed retrieval chain.
    """
    return RetrievalChain(vectorstore, llm)


def print_response(retrieval_chain, prompt):
    """
    Invokes the retrieval chain with a prompt and prints the answer to the console.

    Args:
        retrieval_chain: The retrieval chain to use.
        prompt (str): The question or prompt to send.

    Returns:
        None
    """
    response = retrieval_chain.invoke({"input": prompt})
    print(response["answer"])
