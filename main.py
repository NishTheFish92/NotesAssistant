from src import *
#Testbench
def main():
    user_input = input("Prompt : ")
    llm = init_llm()
    embeddings = init_embeddings()
    output_path = './experiments/output_images'
    #img2txt(output_path,5,llm)
    vectorstore = process_documents(output_path,embeddings)
    print_response(chainer(vectorstore,llm),user_input)
if __name__ == "__main__":
    main()
