# NotesAssistant

## Overview
This project allows you to query professor notes and PPTs using LLMs (Large Language models) through Retrieval Augmented Generation (RAG) with the help of the LangChain framework.

## Features
- Upload professor notes as PPTs (In PDF format) or PDFs and parse them

- Content is embedded and the embeddings are stored in a vector store 

- Query the notes using an LLM

- Is able to understand and interpret images. The user can query the LLM on diagrams.

- Answers from the notes depending on the question.

-  Answers are further enhanced by providing a fun explanation and an intuitive explanation. If the user wants concise answers only from the notes they can add `AOFS` (Answer only from slides) at the end of their query. 

- The vector store is stored locally allowing for persistence
## Tech stack
- Language: Python
- Embedding: BGE-Base V1.5
- Vector store: FAISS
- LLM Provider: Gemini 2.0 Flash
- Frameworks: LangChain

## Getting started
### 1. Clone the repo
```
git clone https://github.com/NishTheFish92/NotesAssistant.git
cd NotesAssistant
```



### 2. Install dependencies
#### UV Installation
```
uv venv
uv sync
```

#### Pip installation

```
pip install -r requirements.txt
```
> **Note:** It is recommended that you have a CUDA enabled system.

### 3. Add your API key
Create a .env file and add your API key
```
GOOGLE_API_KEY=your-key
```

### 4. Add professor notes
Add your PPT (In pdf format) or PDF in the PDFs folder.
### 5. Embedding PDF for querying
Run the train.py file with the arguement as the path to your pdf file.
```
python train.py <path_to_your_pdf>
```
## Usage
Execute the following command
```
python run.py
```
Happy learning!

## Potential Improvements
1. **Multiple PDF support:** The user can choose which PDF they want to query.

2. **Improved output format:** Makes explanations more easier.

3. **Improved summarization:** Allows user to get a better overview regarding the contents of the PDF.

## Known issues
1. The application is unable to summarize the PDF effectively

2. Responses are sometimes longer than required where a a notable part of the response is redundant.

3. The method in which the PDF is embedded is not optimized and consumes a lot of API calls.