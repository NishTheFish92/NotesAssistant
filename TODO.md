# NotesAssistant TODO

Goal: RAG over unstructured PDFs, backed by measured evals. Pipeline is frozen; only the steps below.

## Steps
1. [x] Token chunking (450 BGE tokens) + page number in chunk metadata
2. [x] Answer prompt: notes only, cite `[p. N]`, refuse when not covered. Show pages in CLI/Streamlit. Pin answer model
3. [ ] Golden set on `Mod5.pdf`: 50–80 questions with evidence pages, ~10 unanswerable
4. [ ] Eval script: recall@k, MRR, judged correctness/faithfulness, refusal rates. Compare 4,000-char vs token chunks. Hand-check ~30 judge calls
5. [x] OCR prompt: original + "not colours, fonts or layout" (verbatim transcription gets blocked by Gemini's recitation filter)
6. [ ] README: results, decisions, limitations. Then the resume bullet

## On hold
BM25 hybrid (first if time) · batching · reranker · heading-aware chunks · APIs/MCP · multi-doc
