"""
Chain creation utilities for NotesAssistant.
Links prompts, retrievers, and LLMs for answering questions.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from settings import TOP_K

PROMPT = ChatPromptTemplate.from_template("""
    Answer the question using only the context below, which comes from the user's lecture notes.
    Each part of the context starts with its page number, like [p. 3].
    Cite the page after every claim in the same form, for example [p. 3] or [p. 3, p. 5].
    Do not add facts from your own knowledge.
    If the context does not answer the question, reply exactly: "The notes don't cover this." and nothing else.
    Answer in neatly formatted paragraphs and with Markdown formatting. Do not use backticks in your response.

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
        context = "\n\n".join(f"[p. {doc.metadata['page']}]\n{doc.page_content}" for doc in docs)
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


def source_pages(response):
    """Sorted, de-duplicated page numbers of the chunks used for an answer."""
    return sorted({doc.metadata["page"] for doc in response["context"]})


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
    print("Retrieved pages:", ", ".join(str(p) for p in source_pages(response)))
