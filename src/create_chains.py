from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def chainer(vectorstore,llm):
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question at the end. 
    Answer in paragraphs with no formatting. Headings can be shown with the help of colons.
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

def print_response(retrieval_chain,prompt):
    response = retrieval_chain.invoke({"input":prompt})
    print(response["answer"])