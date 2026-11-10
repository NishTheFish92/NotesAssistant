# NotesAssistant TODO

Goal: RAG over unstructured PDFs, backed by measured evals.
The pipeline (PDF → page images → vision OCR → BGE → FAISS) is frozen. Only the items below; everything else waits.

Done: LangChain 1.x upgrade, `settings.py`, page cache, resumable ingestion, retries, CLI and Streamlit tested.

## Core (in order)
- [ ] Token-based chunking (BGE tokenizer, about 450 tokens) so chunks fit BGE's 512-token limit
- [ ] Page number in every chunk's metadata
- [ ] Answer prompt: notes only by default, cite pages, refuse when the notes don't cover the question
- [ ] Show cited pages in the CLI and Streamlit
- [ ] Pin one model instead of `auto`
- [ ] OCR prompt: faithful transcription, then figures, in Markdown
- [ ] Golden set on `Mod5.pdf`: 50–80 questions labelled with evidence pages, about 10 unanswerable
- [ ] Eval script: recall@k and MRR, LLM-judged correctness and faithfulness, refusal rates (missed and wrong)
- [ ] Hand-label about 30 answers and report judge agreement
- [ ] Compare 4,000-character vs token chunks on recall@k and MRR
- [ ] README: results table, key decisions, limitations. Then the resume bullet

## On hold
- Hybrid BM25 + embeddings (first pick if time is left)
- Vision batching, dpi test, PyMuPDF swap, PPTX input, text-layer hints
- Heading-aware chunking, reranker
- FastAPI, MCP server, new web UI
- Multi-document support, per-page vision routing
- Long-context baseline, judge bias checks, bootstrap CIs, eval in CI, `LEARNING.md`, demo GIF

## Not doing
Code generation · multilingual · fine-tuning · agents
