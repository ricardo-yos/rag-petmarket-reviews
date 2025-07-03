# RAG-Based Conversational Assistant for Pet Market Reviews

RAG-Based Conversational Assistant for Pet Market Reviews is an intelligent chatbot designed to provide insightful and accurate answers based on real customer reviews of pet-related businesses in Santo André, Brazil. Leveraging Retrieval-Augmented Generation (RAG) techniques, the assistant combines semantic search over a custom vector database with advanced language models to deliver context-aware and relevant responses, helping businesses better understand customer feedback and improve their services.

---

## Overview

This project aims to provide an intelligent conversational assistant specialized in the pet care sector. It is designed to:
- Search, filter, and reason over real customer reviews from **pet shops**, **veterinary clinics**, and **grooming services** in Santo André, Brazil.
- Use **semantic search with embeddings** to retrieve the most relevant and meaningful information from the review database.
- Understand and respond to **natural language queries** using a language model capable of generating accurate and human-like answers.
- Support pet business owners and customers by:
	- Identifying strengths and weaknesses based on customer experiences.
	- Highlighting common complaints or praises about specific services or businesses.
	- Helping users compare options based on real feedback rather than just ratings.
- Help with decisions and understanding the market by showing clear information based on what many customers have written in their reviews.
- Enable a more data-informed dialogue between service providers and consumers, enhancing transparency and trust.

---

## Target Audience

- Data scientists and machine learning engineers interested in natural language processing and retrieval-augmented generation.
- Pet shop owners and managers aiming to analyze customer feedback.
- Developers exploring practical implementations of RAG-based conversational agents.
- Researchers focused on applied NLP in business contexts.

---

## Project Structure

```bash
rag_petmarket_reviews/
│
├── data/                         # Central directory for all project data
│   ├── chroma/                   # ChromaDB data storage
│   │   └── chroma.sqlite3        # Main Chroma database file
│   │
│   ├── processed/                # Cleaned and preprocessed data ready for use
│   │   └── places_reviews.json   # JSON file containing processed reviews data
│   │
│   └── raw/                      # Raw data collected from original sources
│       ├── places.csv            # Original data about places (pet shops, clinics, etc.)
│       └── reviews.csv           # Original reviews data
│
├── logs/                         # Directory for log files (initially empty)
│   ├── app.log                   # Logs related to main application (app.py)
│   ├── build_db.log              # Logs for database building process
│   ├── llm_utils.log             # Logs related to LLM utilities and helpers
│   ├── memory.log                # Logs about conversation memory operations
│   └── rag_assistant.log         # Logs from the RAG assistant core logic
│
├── memory/                       # Conversational memory storage
│   └── chat_history.db           # Chat history database file
│
├── src/                          # Source code of the project
│   ├── config/                   # Configuration files and utilities
│   │   ├── __init__.py           # Makes 'config' a Python package
│   │   ├── app_config.yaml       # General application settings (model params, etc.)
│   │   ├── prompt_config.yaml    # Configuration related to prompts/templates
│   │   ├── config_loader.py      # Python module to load YAML config files
│   │   └── paths.py              # Defines important filesystem paths for the project
│   │
│   ├── core/                     # Core logic of the assistant (RAG engine, prompt building, memory)
│   │   ├── __init__.py           # Makes 'core' a Python package
│   │   ├── llm_helpers.py        # Utilities for interacting with the LLM
│   │   ├── memory.py             # Manages conversational memory (loading, saving, updating)
│   │   ├── prompt_builder.py     # Builds and manages language model prompts
│   │   ├── rag_assistant.py      # Main logic for retrieval-augmented generation assistant
│   │   └── rag_loader.py         # Loader for RAG assistant and config
│   │
│   ├── data_processing/          # Data loading, cleaning, and preprocessing scripts
│   │   ├── __init__.py           # Makes 'data_processing' a Python package
│   │   ├── build_db.py           # Script to build or update databases from data sources
│   │   └── generate_json.py      # Script to generate JSON files from raw data
│   │
│   ├── interface/                # User interface code (e.g., Streamlit or API)
│   │   ├── __init__.py           # Makes 'interface' a Python package
│   │   ├── chat_handler.py       # Logic for handling chat input/output in UI
│   │   ├── sidebar.py            # Sidebar UI components and controls
│   │   └── styles/               # Stylesheets and static assets
│   │       └── pet_style.css     # CSS file defining styles for the UI
│   │
│   ├── utils/                    # Utility functions shared across modules
│   │   ├── __init__.py           # Makes 'utils' a Python package
│   │   ├── logger.py             # Logging setup and utilities
│   │   └── translator.py         # Translation utility for multilingual support
│   │
│   └── app.py                   # Main application entry point
│
├── README.md                     # Project overview, usage instructions, and documentation
├── requirements.txt              # List of Python package dependencies
├── .env                          # Environment variables (API keys, paths, secrets)
└── .gitignore                    # Specifies files and folders to ignore in Git version control
```

---

## Prerequisites

- Python version 3.10 or newer installed on your system.
- You need to have a valid `GROQ_API_KEY` set in your environment variables to access the required APIs.
- A computer with a GPU is recommended to speed up model processing, but not mandatory.

---

## Installation

Make sure you have Python 3.10 or higher installed.

### 1. Clone the repository

Download the project to your local machine using Git:

```bash
git clone https://github.com/ricardo-yos/rag-petmarket-reviews.git
cd rag-petmarket-reviews
```

### 2. Create a virtual environment

Set up a dedicated Conda environment for the project:

```bash
conda create -n rag_env python=3.10
conda activate rag_env
```

### 3. Install dependencies

Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

- Create a `.env` file at the project root and add your API keys or environment variables, for example:

```bash
GROQ_API_KEY=your_api_key_here
```

- Review and customize configuration files under `src/config/` as needed.

---

## Usage

### 1. Prepare the data

- `generate_json.py`: Integrates data from places and reviews into a unified JSON file, associating each place with its corresponding reviews.
- `build_db.py`: Creates chunked review texts and generates their embeddings, then stores them in a ChromaDB collection for semantic search.

```bash
cd src
python -m data_processing.generate_json       # Merge place and review data into JSON
python -m data_processing.build_db            # Chunk reviews and store embeddings in ChromaDB
cd ..                                         # Return to the project root
```

### 2. Launch the application

Start the assistant using **Streamlit**:

```bash
PYTHONPATH=$(pwd)/src streamlit run src/app.py
```
> Make sure you're in the project root (`rag-petmarket-reviews/`) when running this command, so Python can locate the correct modules via `PYTHONPATH`.
>
> This will open a **web interface in your browser** where you can interact with the assistant via natural language.

### 3. User Interface (Streamlit)

The application uses Streamlit to deliver a simple and intuitive chat interface:

- Ask questions in **any language** about pet-related services.
- The assistant always responds in the **same language** as the user's question.
- It retrieves the most relevant **customer reviews** using **semantic search** and responds strictly based on that data.
- A **sidebar** allows you to configure the similarity threshold and number of documents retrieved.
- **Conversational memory** helps the assistant maintain context across multi-turn interactions.

**Example query**:

> ❓ "What are the main complaints about grooming services?"
>
> 💬 "Based on customer reviews:
> - "Delays in starting the grooming service"
> - "Price considered too high for the quality of the service"
> - "Lack of communication about deadlines or the pet's status"

---

## Data Requirements

The project expects the following data formats and setup:

- **Input files**: Located in `data/raw/`:
  - `places.csv`: Contains metadata about pet-related businesses.
  - `reviews.csv`: Contains customer reviews linked by `place_id`.

- **Processed data**: Combined dataset generated by merging `places.csv` and `reviews.csv`, saved as `data/processed/places_reviews.json`. Containing both business information and associated reviews.

### Example structure

```json
[
  {
    "place_id": "abc123xyz",
    "name": "Pet Love Center",
    "street": "Rua das Flores, 123",
    "neighborhood": "Jardim Primavera",
    "city": "Santo André",
    "rating": 4.5,
    "num_reviews": 25,
    "type": "pet_store",
    "latitude": -23.000000,
    "longitude": -46.000000,
    "reviews": [
      {
        "review_id": "joana_silva_1729000000",
        "author": "Joana Silva",
        "rating": 5,
        "text": "Excelente atendimento! Minha cachorrinha foi muito bem cuidada no banho e tosa.",
        "review_length": 100,
        "word_count": 17,
        "time": 1729000000,
        "date": "2024-10-15 14:30:00",
        "response": "pt"
      }
    ]
  }
]
```

- **Vector database**: Embedding index generated with a multilingual MiniLM model (`paraphrase-multilingual-MiniLM-L12-v2`), stored in `data/chroma/`.

Make sure to run the data processing scripts before launching the app to ensure all required files are in place.

---

## Configuration

The project uses configuration files to centralize and simplify the management of key settings.

### 1. `.env` File

Located at the project root (`rag-petmarket-reviews/.env`), this file contains environment variables such as API keys.

Example:

```bash
GROQ_API_KEY=your_api_key_here
```

This file is **not included in version control** (.gitignore) and must be created manually.

### 2. `app_config.yaml`

Path: `src/config/app_config.yaml`

Defines the app settings used by the RAGAssistant, such as:

- **LLM model**: Defines the language model used (`llama-3.1-8b-instant`, via Groq or another provider)
- **Vector search**: Number of results and cosine similarity threshold for semantic retrieval
- **Memory management**: How much recent chat history is kept or summarized
- **Reasoning strategies**: Predefined step-by-step thinking templates for generating responses

**Reasoning Strategies**

Defined under `reasoning_strategies`, these templates structure how the assistant reasons through complex questions:

- **CoT (Chain of Thought)**
  - Break the problem into logical steps before giving the answer. (default)

- **ReAct (Reason + Act)**
  - Uses a cycle of *Thought → Action → Observation → Reflection* to iteratively build the answer.

- **Self-Ask**
  - Breaks the question into sub-questions, answers each one, and combines them into a final response.

To change the default strategy:

```yaml
llm: "llama-3.1-8b-instant"

vectordb:
  threshold: 0.3
  n_results: 5

memory_strategies:
  trimming_window_size: 6
  summarization_max_tokens: 1000

reasoning_strategies:
  default: "CoT"
```

### 3. `prompt_config.yaml`

Path: `src/config/prompt_config.yaml`

Defines how prompts are constructed for the language model. This includes:

- **Role and tone**: Tells the model it’s a review-based assistant for pet services
- **Instructions**: Ensures the assistant only uses retrieved reviews to answer
- **Constraints**: Prevents speculation, external knowledge, or breaking character
- **Output format**: Enforces Markdown and concise answers

Example snippet:

```yaml
prompt:
  system_message: |
    You are a helpful assistant that answers questions based on customer reviews.
  user_template: |
    Based on the reviews, answer the following question: {question}
```

### 4. paths.py

Path: `src/config/paths.py`

Centralized Python module that defines key file system paths used throughout the project, such as:
- Path to raw and processed data
- Vector DB storage
- Chat history DB

Keeping all path definitions here allows for easier refactoring and consistency across modules.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

## Contributing

Contributions are welcome and appreciated! If you'd like to help improve this project, please follow the steps below:

### How to Contribute

1. **Fork the repository**  
Create your own copy of the project on GitHub.

2. **Clone your fork locally and make changes**  
Clone your fork to your computer, create a new branch, and implement your changes.

3. **Submit a pull request**  
Send a pull request with a detailed description of your modifications.

---

### Contact

Feel free to reach out via GitHub for questions, feedback, or collaboration opportunities.
