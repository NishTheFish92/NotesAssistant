# NotesAssistant TODO

Goal: RAG over any unstructured PDF (slides, notes, scans, books), with measured quality.
Pipeline stays straightforward: every page goes through the vision model.

## 1. Fixes and foundations
Code changes below are done but **not yet run**: they need `uv sync` first.
- [ ] `uv sync`, then test `index.py`, `run_cli.py` and `streamlit run app.py`
- [ ] Re-index `Mod5.pdf` if the old `faiss_index` won't load under the new LangChain (87 vision calls, so maybe after batching and caching)
- [x] Upgrade to LangChain 1.x and switch to current import paths (`create_retrieval_chain` replaced with a small chain in `src/create_chains.py`)
- [x] Make the Streamlit app importable from the repo root (moved to `app.py`)
- [x] Build the chain once in the CLI, not on every question
- [x] Fix the image type: pages are saved as JPEG but sent labelled PNG
- [x] Add `allow_dangerous_deserialization` to `FAISS.load_local` (wrapped in `load_vectorstore`)
- [x] Move all settings into one config (`settings.py`): dpi, paths, models, chunk size, top-k, batch size
- [x] Regenerate `requirements.txt` from `uv export`
- [x] Empty `output_folder` before each run, so pages from a previous, longer PDF don't leak into the new index
- [x] Fix wrong README commands (`uv sync`, `streamlit run app.py`, `python run_cli.py`, FreeLLMAPI env vars)
- [x] Rename `train.py` to `index.py`
- [x] Use BGE's documented query prefix for search

## 2. Ingestion
- [ ] Extract the text layer for every page
- [ ] OCR prompt: transcribe the text faithfully, then describe the figures
- [ ] Send the text layer as a hint next to each image
- [ ] Batch about 5 pages per vision call, with results keyed by page number
- [x] Cache each page's result (file hash + page + prompt/model/dpi hash), so re-runs make 0 calls
- [x] Make ingestion resumable, so a failed batch doesn't restart the whole document
- [x] One retry layer for vision calls: retry rate limits, timeouts and 5xx, fail fast on 503 and 4xx
- [ ] Test a lower dpi (120–150 vs 200)
- [ ] Optional: PPTX input (convert to PDF with LibreOffice)

## 3. Chunking and retrieval
- [ ] Add metadata to every chunk: page number, and heading if detectable (needed for citations)
- [ ] Structure-aware chunking: pages and headings, with a size cap
- [ ] Add a keyword index (BM25)
- [ ] Hybrid retrieval: keyword plus embeddings, merged with reciprocal rank fusion
- [ ] Cross-encoder reranker, kept only if evals show a gain
- [ ] Make top-k configurable

## 4. Answers
- [ ] Cite the document and page in every answer
- [ ] Refuse when the documents don't cover the question
- [ ] Prompt modes: strict "from the notes only" as the default, plus an "explain with analogies" option
- [ ] Pin one model for evals instead of `auto`

## 5. Interfaces
- [ ] FastAPI endpoints: `/ingest`, `/search`, `/ask`
- [ ] Web UI with PDF upload and clickable citations
- [ ] MCP server with `search_docs` and `ask` tools
- [ ] Working CLI

## 6. Evals
- [ ] Choose a fixed eval corpus: `Mod5.pdf` (messy slides) plus one large unstructured PDF too big for a single prompt
- [ ] Write a golden set of 100–150 questions yourself: factual, explanatory, figure-based, multi-page and unanswerable
- [ ] Label each question's evidence by document and page, not chunk ID
- [ ] Freeze a dev/test split before any tuning
- [ ] Ingestion eval: hand-transcribe about 10 pages and compare text-layer vs vision vs vision-with-text-hint extraction against them
- [ ] Retrieval metrics: recall@k and MRR, broken down by question type
- [ ] Answer metrics: correctness, faithfulness, citation accuracy, and both refusal rates (missed refusals and wrong refusals)
- [ ] LLM judge, checked against 50 answers you label by hand (agreement % and Cohen's kappa)
- [ ] Check the judge for verbosity and position bias
- [ ] Cost and speed: ingestion calls and time per page before and after batching, and query latency
- [ ] Long-context baseline (the "why RAG?" answer)
- [ ] Paired comparisons and bootstrap confidence intervals, so you know when a difference is noise
- [ ] Run all evals with one command, and run the retrieval eval in CI
- [ ] Keep an experiment log, including changes that didn't help
- [ ] Read 20+ wrong answers per run and tag their failure types

## 7. Write-up
- [ ] README: architecture diagram, key decisions, results table, "what failed and why", limitations, and correct setup commands
- [ ] Demo GIF
- [ ] `LEARNING.md`: one line per concept learned and per decision
- [ ] Resume bullet with only measured numbers
- [ ] A 5-minute story about one failure you found and fixed

## Later (experiments)
- [ ] Multi-document support: a folder per document keyed by a hash of the file (fixes the overwrite bug), document name in chunk metadata, one index holding all documents with filtering by document
- [ ] Per-page vision routing (text-layer length, image area, vector drawings) vs vision on every page, compared on the ingestion eval

## Not doing
Code generation · multilingual · fine-tuning · agents
