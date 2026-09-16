# Document RAG Assistant

This project is a document question answering application built with:

- Streamlit for the user interface.
- FastAPI for the backend API.
- Milvus for storing document embeddings and running similarity search.
- OpenRouter embeddings for converting document text and questions into vectors.
- NVIDIA's model endpoint for generating answers.
- LangChain for document processing and retrieval orchestration.

The application accepts PDF, DOCX, CSV, XLS, and XLSX files. It extracts the text, splits text documents into chunks, stores the chunks and their embeddings in a selected Milvus collection, and answers questions using the most similar chunks.

## Project flow

```text
User uploads a document in Streamlit
        |
        v
FastAPI /upload
        |
        v
DocumentProcessor extracts and chunks the content
        |
        v
OpenRouter creates embeddings
        |
        v
Milvus stores the chunks in the selected collection

User asks a question in Streamlit
        |
        v
FastAPI /search
        |
        v
OpenRouter creates a query embedding
        |
        v
Milvus returns similar chunks
        |
        v
NVIDIA generates the final answer
```

## Requirements

Install these tools before starting:

- Python 3.11 or later
- Docker and Docker Compose
- An OpenRouter API key
- An NVIDIA API key

The document parser uses pandas for CSV and Excel files. If pandas or Excel support is not installed in your environment, install them with:

```bash
pip install pandas openpyxl
```

## 1. Create and activate a virtual environment

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install Python packages

```bash
pip install -r requirements.txt
pip install pandas openpyxl
```

## 3. Configure API keys

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
NVIDIA_API_KEY=your_nvidia_api_key
```

Do not commit `.env`. It is already excluded by `.gitignore`.

The current default configuration is:

| Setting | Value |
|---|---|
| Milvus URI | `http://localhost:19530` |
| Chat model | `nvidia/nemotron-3-super-120b-a12b` |
| Embedding model | `nvidia/nemotron-3-embed-1b:free` |
| Chunk size | `1000` characters |
| Chunk overlap | `200` characters |
| Default collection | `MyLangChainCollection` |

These values are defined in [config.py](config.py).

## 4. Start Milvus

Start Milvus and its required services from the project directory:

```bash
docker compose up -d
```

Check the containers:

```bash
docker compose ps
```

Milvus is available at `http://localhost:19530`. Attu, the optional Milvus web interface, is available at [http://localhost:3001](http://localhost:3001).

To stop the services:

```bash
docker compose down
```

To stop them and remove stored Milvus data:

```bash
docker compose down -v
```

The last command deletes the Docker volumes containing Milvus data.

## 5. Start the FastAPI backend

Keep Milvus running and open a terminal in the project directory:

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` first.

The backend is now available at [http://localhost:8000](http://localhost:8000). FastAPI's interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## 6. Start the Streamlit interface

Open a second terminal, activate the same environment, and run:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Streamlit normally opens at [http://localhost:8501](http://localhost:8501).

The Streamlit application expects the FastAPI backend at `http://localhost:8000`. Change `API_URL` at the top of [streamlit_app.py](streamlit_app.py) if the backend runs elsewhere.

## 7. Upload and query a document

1. Open the Streamlit URL.
2. Select a Milvus collection from the **Document collection** dropdown.
3. Select a PDF, DOCX, CSV, XLS, or XLSX file.
4. Click **Upload Document**.
5. Wait for the indexing success message.
6. Keep the same collection selected when asking questions.
7. Enter a question in the chat box.

The selected collection is used both when uploading the document and when searching for answers.

The current dropdown contains:

- `MyLangChainCollection`
- `documents`

These options are defined in `streamlit_app.py`. The selected Milvus collection is created when the first document is inserted if it does not already exist.

## API usage

### Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf" \
  -F "collection_name=MyLangChainCollection"
```

### Search documents

```bash
curl -X POST "http://localhost:8000/search?user_query=What%20is%20this%20document%20about%3F&top_k=8&collection_name=MyLangChainCollection"
```

`top_k` is limited to a maximum of 30 by the API.

### Direct structured output demo

```bash
curl -X POST "http://localhost:8000/invoke?user_query=Give%20me%20a%20movie%20recommendation"
```

The `/invoke` and `/stream` routes are standalone LLM demos and do not use the uploaded documents.

## Main files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web interface for uploading files and asking questions. |
| `main.py` | FastAPI endpoints for upload, search, and direct LLM calls. |
| `config.py` | Models, Milvus connection, chunking, and retrieval settings. |
| `services/document_parser.py` | Reads supported file types and creates LangChain documents. |
| `services/vector_db.py` | Creates the Milvus vector store and performs insert and similarity search operations. |
| `services/retrieval_service.py` | Connects API requests to the selected vector store. |
| `services/AI_models.py` | Creates the chat and embedding model clients. |
| `services/llm_service.py` | Builds the RAG prompt chain and handles rate-limit retries. |
| `docker-compose.yml` | Runs Milvus, etcd, MinIO, and Attu. |
| `Others/` | Schema migration scripts, examples, and reference notes. |

## Supported file behavior

- PDF files are read page by page and split into chunks.
- DOCX files are read from their non-empty paragraphs and split into chunks.
- CSV files are converted into one document per non-empty row.
- XLS and XLSX files are read sheet by sheet and converted into row documents.
- Each document stores source metadata such as file name, page number, sheet name, row number, and document ID.

## Troubleshooting

### Milvus connection refused

Make sure Docker is running and Milvus is healthy:

```bash
docker compose ps
docker compose logs milvus
```

### Backend cannot start because of missing packages

Activate the project virtual environment and reinstall dependencies:

```bash
pip install -r requirements.txt
pip install pandas openpyxl
```

### No answer is returned

Check that:

1. The backend is running on port 8000.
2. A document was uploaded successfully.
3. The same collection is selected for upload and search.
4. The selected API keys are present in `.env`.
5. The OpenRouter and NVIDIA services are reachable.

### API key errors

Verify that `.env` is in the project root and contains non-empty values for both `OPENROUTER_API_KEY` and `NVIDIA_API_KEY`. Restart the FastAPI process after changing `.env`.

## Complete startup checklist

From a fresh checkout:

```bash
cd /path/to/LangChainProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pandas openpyxl
```

Create `.env`, then start the services in separate terminals:

```bash
# Terminal 1
docker compose up -d

# Terminal 2
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open Streamlit, select a collection, upload/select collaction, and ask a question.
