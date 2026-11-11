"""
Retrieval eval: hit@k, MRR and recall@k on the golden set. No LLM calls.

Run from the repo root:
    python -m eval.retrieval_eval            # saved index (current chunking)
    python -m eval.retrieval_eval --chars    # 4,000-char baseline, built in memory

A retrieved chunk counts as its page. Unanswerable questions are skipped.
"""

import argparse
import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import INDEX_PATH, OUTPUT_PATH, TOP_K
from src import init_embeddings, load_vectorstore

GOLDEN_PATH = "eval/golden_set.jsonl"


def char_vectorstore(embeddings):
    """The old character split (4,000 / 200), built from the extracted page text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    text_dir = os.path.join(OUTPUT_PATH, "output_text")
    docs = []
    for file_name in sorted(f for f in os.listdir(text_dir) if f.endswith(".txt")):
        with open(os.path.join(text_dir, file_name), encoding="utf-8") as f:
            text = f.read()
        page = int(file_name.removeprefix("page_").removesuffix(".txt"))
        docs += [Document(page_content=c, metadata={"page": page}) for c in splitter.split_text(text)]
    return FAISS.from_documents(docs, embeddings)


def main():
    parser = argparse.ArgumentParser(description="Score retrieval on the golden set.")
    parser.add_argument("--chars", action="store_true", help="use the 4,000-char baseline")
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    embeddings = init_embeddings()
    vectorstore = char_vectorstore(embeddings) if args.chars else load_vectorstore(INDEX_PATH, embeddings)
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]
    golden = [q for q in golden if q["pages"]]

    by_type = {}
    misses = []
    for q in golden:
        pages = [d.metadata["page"] for d in vectorstore.similarity_search(q["question"], k=args.k)]
        evidence = set(q["pages"])
        rank = next((i + 1 for i, p in enumerate(pages) if p in evidence), None)
        hit = rank is not None
        if not hit:
            misses.append((q["id"], q["pages"], pages))
        row = by_type.setdefault(q["type"], {"n": 0, "hit": 0, "rr": 0.0, "recall": 0.0})
        row["n"] += 1
        row["hit"] += hit
        row["rr"] += 1 / rank if hit else 0
        row["recall"] += len(evidence & set(pages)) / len(evidence)

    print(f"\n{'4,000-char' if args.chars else 'saved index'} chunks, k = {args.k}, {vectorstore.index.ntotal} chunks\n")
    print(f"{'type':<10}{'n':>4}{'hit@k':>8}{'MRR':>7}{'recall@k':>10}")
    total = {"n": 0, "hit": 0, "rr": 0.0}
    for t, r in sorted(by_type.items()):
        recall = f"{r['recall'] / r['n']:.2f}" if t == "multi" else "-"
        print(f"{t:<10}{r['n']:>4}{r['hit'] / r['n']:>8.2f}{r['rr'] / r['n']:>7.2f}{recall:>10}")
        for key in total:
            total[key] += r[key]
    print(f"{'all':<10}{total['n']:>4}{total['hit'] / total['n']:>8.2f}{total['rr'] / total['n']:>7.2f}{'-':>10}")

    print(f"\nMisses ({len(misses)}): id, evidence pages, retrieved pages")
    for m in misses:
        print(*m)


if __name__ == "__main__":
    main()
