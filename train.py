from src import *
import PyPDF2
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train NotesAssistant with a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_path = './output_images'
    num_pages = 0
    with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

    toimages(pdf_path,output_path)
    llm = init_llm()
    embeddings = init_embeddings()
    img2txt(output_path, num_pages, llm)

    vectorstore = embed_documents(output_path, embeddings)
    vectorstore.save_local("faiss_index")

    print("PPT indexed. Vector store saved to 'faiss_index'.")

if __name__ == "__main__":
    main()