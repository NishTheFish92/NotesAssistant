import argparse

from settings import EMBED_BATCH_SIZE, INDEX_PATH, OUTPUT_PATH
from src import embed_documents, img_to_txt, init_embeddings, init_llm, pdf_to_images


def main():
    parser = argparse.ArgumentParser(description="Index a PDF for NotesAssistant.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    num_pages = pdf_to_images(args.pdf_path, OUTPUT_PATH)
    llm = init_llm()
    embeddings = init_embeddings()
    print("This will take a while...")
    img_to_txt(OUTPUT_PATH, num_pages, llm)
    vectorstore = embed_documents(OUTPUT_PATH, embeddings, EMBED_BATCH_SIZE)
    vectorstore.save_local(INDEX_PATH)

    print(f"PDF indexed ({num_pages} pages). Vector store saved to '{INDEX_PATH}'.")


if __name__ == "__main__":
    main()
