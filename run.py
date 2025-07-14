from langchain.vectorstores import FAISS
from src import init_embeddings,init_llm
from src import chainer,print_response

embeddings = init_embeddings()
llm = init_llm()
vectorstore = FAISS.load_local("faiss_index",embeddings)
print("\n\n")
print("To exit the program type 0 in the query section.\n")
while(True):
    user_input = input("Ask away! :- ")
    if(user_input == "0"):
        break
    rchain = chainer(vectorstore,llm)
    print_response(rchain,user_input)

