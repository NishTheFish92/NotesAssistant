from src import pdf_to_images,init_embeddings,init_llm,img_to_txt,embed_documents
import PyPDF2
import argparse
from settings import OUTPUT_PATH

def main():
    parser = argparse.ArgumentParser(description="Train NotesAssistant with a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_path = OUTPUT_PATH
    num_pages = pdf_to_images(pdf_path,output_path)
    llm = init_llm()
    embeddings = init_embeddings()
    img_to_txt(output_path, num_pages, llm)
    vectorstore = embed_documents(output_path, embeddings)
    vectorstore.save_local("faiss_index")

    print("PPT indexed. Vector store saved to 'faiss_index'.")

if __name__ == "__main__":
    main()