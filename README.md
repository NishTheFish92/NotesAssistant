# NotesAssistant

## Overview
This project allows you to query professor notes and PPTs using LLMs (Large Language models) through Retrieval Augmented Generation (RAG) with the help of the LangChain framework.

## Features
- Upload professor notes as PPTs (In PDF format) or PDFs and parse them.

- Content is embedded and the embeddings are stored in a vector store .

- Query the notes using an LLM.

- Is able to understand and interpret images. The user can query the LLM on diagrams.

- Answers from the notes depending on the question.

-  Answers are further enhanced by providing a fun explanation and an intuitive explanation. If the user wants concise answers only from the notes they can add `AOFS` (Answer only from slides) at the end of their query. 

- The vector store is stored locally allowing for persistence.

- Streamlit powered user interface, enhancing readability of responses and ease of prompting.
## Tech stack
- Language: Python
- Embedding: BGE-Base V1.5
- Vector store: FAISS
- LLM Provider: any OpenAI-compatible endpoint (FreeLLMAPI by default)
- Frameworks: LangChain

## Getting started
### 1. Clone the repo
```
git clone https://github.com/NishTheFish92/NotesAssistant.git
cd NotesAssistant
```



### 2. Install dependencies
#### Install via UV
```
uv sync
```

#### Install using pip

```
pip install -r requirements.txt
```
On Windows, add `--extra-index-url https://download.pytorch.org/whl/cu130` so pip can find the CUDA build of PyTorch.

> **Note:** It is recommended that you have a CUDA enabled system. `pdf2image` also needs Poppler installed (`sudo apt install poppler-utils` on Debian/Ubuntu).

### 3. Add your API key
Create a .env file with your FreeLLMAPI key and endpoint (the URL defaults to `http://localhost:3001/v1` if omitted)
```
FREELLM_API_KEY=your-key
FREELLM_BASE_URL=http://localhost:3001/v1
```

### 4. Embedding PDF for querying
Run index.py with the path to your PDF file as the argument.
```
python index.py <path_to_your_pdf>
```
## How to query
### Using streamlit
Execute the following command
```
streamlit run app.py
```

### CLI interface

Execute the following command
```
python run_cli.py
```
Happy learning!

## Evaluation
The pipeline is scored against a hand-written golden set (`eval/golden_set.jsonl`) of 58 questions on one deck of lecture slides on I/O organisation:

| Type | Count | What it tests |
|---|---|---|
| single | 36 | answer is on one slide |
| multi | 8 | answer spans several slides |
| numeric | 4 | worked problems from the slides |
| trap | 8 | on topic, but the slides don't cover it |
| off_topic | 2 | different subject entirely |

Trap and off-topic questions should get the refusal "The notes don't cover this."

### Running it
Run these from the repo root:
```
python -m eval.retrieval_eval      # hit@k, MRR, recall@k only (no LLM calls)
python -m eval.answer_eval         # answer every question, then have a judge LLM grade it
python -m eval.score               # print the metrics from eval/results.jsonl (no LLM calls)
```
`answer_eval` writes to `eval/results.jsonl` after every question, so a run that crashes can be resumed. The judge model (`JUDGE_MODEL` in `settings.py`) must be different from the answer model.

To check the judge, run `python -m eval.score --sample 30`. Fill in the `human` field in `eval/judge_audit.json`, then run `python -m eval.score` again to see the agreement and Cohen's kappa.

### Metrics
- **hit@k / MRR / recall@k**: was an evidence slide in the top k chunks, how high was it ranked, and what share of the evidence slides were retrieved.
- **correct**: the answer has the key facts of the reference and nothing that contradicts it.
- **faithful**: every claim in the answer is in the retrieved chunks.
- **wrong / missed refusals**: refusing an answerable question, or answering an unanswerable one.
- **citations**: whether the `[p. N]` page tags point to retrieved slides and to evidence slides.

### Results
Answer model `gemini-3.5-flash-lite`, judge `nemotron-3.5-lightning-30b-a3b`, k = 4, chunks of 450 tokens:

| Metric | Score |
|---|---|
| hit@4 | 45/48 (94%) |
| MRR | 0.88 |
| recall@4 | 0.88 |
| correct | 42/45 (93%) |
| faithful | 46/46 (100%) |
| wrong refusals | 3/48 (6%) |
| missed refusals | 1/10 (10%) |
| citations to a retrieved slide | 173/173 (100%) |
| citations to an evidence slide | 152/171 (89%) |

The judge's verdicts were reviewed by hand. Five were corrected, and three golden references that asked for more than their question did were trimmed. The remaining failures are mostly cases where the slide with the needed fact wasn't retrieved (q03, q18, q28, q29, q32) and one over-strict refusal on a question that combines several slides (q48).

## Improvements
- Implemented batching while embedding large amount of text. Changing the VRAM usage from ~4GB to <1GB
- Added linux compatibility.
- Added an interactive user interface to allow users to query PPTs with more ease. User interface designed using streamlit. 

## Potential Improvements
1. **Multiple PDF support:** The user can choose which PDF they want to query.

2. <s> **Improved output format:** Makes explanations more easier. </s>

3. **Improved summarization:** Allows user to get a better overview regarding the contents of the PDF.

## Known issues
1. The application is unable to summarize the entire PDF effectively

2. Responses can be longer than required where a a notable part of the response is redundant.

3. The method in which the PDF is embedded is not optimized and consumes a lot of API calls.
