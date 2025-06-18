# Sentiment Analysis API

This project processes e-commerce review data, generates embeddings, stores them in a vector database, and exposes a sentiment analysis API using FastAPI.

## Structure
- `src/`: Source code modules (API, logic, utilities)
- `data/`: Data and database files (auto-created with process_data.py)
- `process_data.py`: Data processing and storing in ChromaDB

## Setup

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Rename `.env.example` to `.env` and add your own API keys:

## Data Processing

Run the following to process data and populate ChromaDB:
```bash
python process_data.py
```

## Running the API

Start the FastAPI server:
```bash
python src/api.py
```

## API Testing

- By default, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser for interactive Swagger documentation.
- If you change the port when running the API, adjust the link accordingly (e.g., `http://127.0.0.1:9000/docs`).
- Example endpoint:
  ```
  GET /analyze_sentiment?product_id=YOUR_PRODUCT_ID
  ```

## Modules

- `data_processing.py`: Data loading and cleaning
- `embedding.py`: Embedding generation
- `database.py`: ChromaDB interaction
- `sentiment_analyzer.py`: Sentiment analysis logic
- `api.py`: FastAPI app
