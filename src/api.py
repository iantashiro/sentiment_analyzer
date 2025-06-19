import sys
import os
# Ensure project root is in sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
import json
from src.sentiment_analyzer import sentiment_analyzer, configure_gemini
from src.database import get_chromadb_client, get_or_create_collection

# Load API keys from environment variables or a .env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
DB_PATH = os.getenv('DB_PATH', 'data/review_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'olist_reviews')
PRODUCT_STATS_PATH = os.getenv('PRODUCT_STATS_PATH', 'data/product_stats.json')

# Load product stats
try:
    with open(PRODUCT_STATS_PATH, 'r', encoding='utf-8') as f:
        product_stats_data = json.load(f)
except FileNotFoundError:
    product_stats_data = {}

# Initialize ChromaDB and Gemini model
client = get_chromadb_client(DB_PATH)
collection = get_or_create_collection(client, COLLECTION_NAME)
gen_model = configure_gemini(GOOGLE_API_KEY)

app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze sentiment of product reviews using a LLM.",
    version="1.0.0"
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Go to /docs to see the available endpoints."}

@app.get("/analyze_sentiment")
async def analyze_sentiment_endpoint(product_id: str):
    if not gen_model or not collection:
        raise HTTPException(status_code=503, detail="AI Service or Database not available. Check server logs.")
    analysis_result = sentiment_analyzer(product_id, collection, gen_model, product_stats_data)
    return analysis_result

if __name__ == "__main__":
    import uvicorn
    print("API running at: http://127.0.0.1:8000")
    print("Swagger docs:   http://127.0.0.1:8000/docs")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
