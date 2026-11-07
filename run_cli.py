"""
Run this file to use a CLI version of NotesAssistant.
"""

from settings import INDEX_PATH
from src import chainer, init_embeddings, init_llm, load_vectorstore, print_response


def main():
    embeddings = init_embeddings()
    llm = init_llm()
    vectorstore = load_vectorstore(INDEX_PATH, embeddings)
    rchain = chainer(vectorstore, llm)

    print("\n\nTo exit the program type 0 in the query section.\n")
    while True:
        user_input = input("Ask away! :- ")
        if user_input == "0":
            break
        print_response(rchain, user_input)


if __name__ == "__main__":
    main()
