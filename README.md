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
