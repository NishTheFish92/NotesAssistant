from src import *
import PyPDF2
def main():
    pdf_path = './PDFs/Mod5.pdf'  # Change as needed
    output_path = './output_images'
    num_pages = 20  
    #with open('../PDFs/Mod5.pdf', "rb") as f:
    #        reader = PyPDF2.PdfReader(f)
    #        num_pages = len(reader.pages)

    toimages(pdf_path,output_path)
    llm = init_llm()
    embeddings = init_embeddings()
    img2txt(output_path, num_pages, llm)

    vectorstore = embed_documents(output_path, embeddings)
    vectorstore.save_local("faiss_index")

    print("PPT indexed. Vector store saved to 'faiss_index'.")

if __name__ == "__main__":
    main()