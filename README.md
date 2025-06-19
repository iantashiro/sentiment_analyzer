# AI-Powered Sentiment Analysis API

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Framework](https://img.shields.io/badge/framework-FastAPI-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## Business Problem & Solution

In today's competitive e-commerce landscape, understanding customer feedback at scale is crucial for business success. Companies receive thousands of reviews, making it impossible for teams to manually read and categorize them efficiently. This leads to missed opportunities, delayed responses to critical issues, and a poor understanding of customer satisfaction.

This project solves this problem by providing an intelligent, automated service that empowers business teams by:

- **Delivering immediate, data-driven insights** from thousands of customer reviews, eliminating the need for manual analysis.
- **Providing a quantitative overview** with statistically sound metrics like average score and total review count.
- **Offering qualitative depth** by summarizing the overall customer narrative and extracting specific points of praise and concern.
- **Enabling easy integration** into any dashboard or internal tool via a fast and reliable REST API.

## Architecture Overview

The project's architecture is divided into two distinct stages: a **one-time data processing pipeline** that builds our knowledge base, and a **real-time API request lifecycle** that uses this knowledge base to generate insights.

### Stage 1: The Data Ingestion & Indexing Pipeline (One-Time Setup)

This initial, one-time process is responsible for creating the foundation of our entire system. To ensure code quality and maintainability, this pipeline is separated into distinct, single-responsibility scripts (`process_data.py`, etc).

1.  **Data Ingestion & Preprocessing:** The process starts by downloading the raw Olist e-commerce dataset. The review data is cleaned, and relevant text fields (like review title and message) are merged to create a single, comprehensive document for each review.
2.  **Embedding Generation:** The cleaned review texts are then converted into semantic vector embeddings using a pre-trained `sentence-transformer` model. These embeddings are numerical representations that capture the meaning of the text.
3.  **Vector Storage:** The generated embeddings and their associated metadata (`product_id`, `review_score`, etc.) are indexed and stored in a persistent **ChromaDB** database. This database is optimized for fast semantic searches.
4.  **Statistical Pre-computation:** To ensure data integrity and avoid sample bias, aggregate statistics (like the true average score and total review count for every product) are pre-calculated from the *entire* original dataset and saved to a simple lookup file, `product_stats.json`.

### Stage 2: The API Request Lifecycle (Real-Time RAG)

When a user makes a request to the `/analyze_sentiment` endpoint, the following **Retrieval-Augmented Generation (RAG)** process is triggered:

1.  **Retrieval:** The system performs two parallel retrievals for the given `product_id`:
    * It queries the **ChromaDB** database to fetch the *texts* of relevant reviews.
    * It looks up the `product_id` in the `product_stats.json` file to get the pre-computed `average_score` and `review_count`. This crucial step ensures our statistical data is unbiased and based on all reviews, not just those with text.

2.  **Augmentation:** The retrieved review texts and the product's average score are dynamically injected into a carefully engineered prompt, providing the Large Language Model (LLM) with context.

3.  **Generation:** The augmented prompt is sent to the **Google Gemini 2.5 flash** model, which generates the qualitative analysis (summary, positive/negative points). This AI-generated result is then combined with the statistical data from the retrieval step to form the final, comprehensive JSON response.

---

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
python main.py
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
