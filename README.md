# PolicyPilot — Evidence-Based AI Policy Assistant

PolicyPilot is a local AI assistant that answers company policy questions using Retrieval-Augmented Generation (RAG) and an LLM-based tool-selection agent.

## Features

- **AI tool selection:** Llama 3 selects a tool based on the user's request.
- **Semantic search:** ChromaDB retrieves relevant policy sections using text embeddings.
- **Local AI inference:** Ollama runs the language models locally.
- **Evidence-backed answers:** Policy responses include supporting text retrieved from the policy document.
- **Python tools:** Supports addition and multiplication.

## Technology Stack

- Python
- Ollama and Llama 3
- nomic-embed-text
- ChromaDB
- Requests
- Retrieval-Augmented Generation (RAG)
- Streamlit
## Project Structure

```text
PolicyPilot/
├── AG.py
├── RAG.py
├── company_policy.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.12
- Ollama installed and running
- The `llama3` and `nomic-embed-text` models downloaded

## Installation

### 1. Download the models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the assistant

```bash
python AG.py
```

Enter a company policy question when prompted.

## How It Works

1. The user submits a question.
2. Llama 3 selects an available tool.
3. Python executes the selected tool.
4. For policy questions, the RAG pipeline embeds the question and retrieves relevant policy sections from ChromaDB.
5. Llama generates an answer using the retrieved information.
6. The assistant displays the answer followed by its supporting evidence.

## Limitations

This is a learning project and is not intended for critical HR or legal decisions. Tool selection and generated answers can be incorrect, and retrieved evidence may be incomplete or irrelevant. Verify policy answers against the original document.

## Future Improvements

- Automated tests and retrieval evaluation
- More precise evidence selection
- Improved tool validation and error handling
- A web-based user interface
