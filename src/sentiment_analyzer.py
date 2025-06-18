import json
import google.generativeai as genai
from datetime import datetime

def configure_gemini(api_key, model_name='gemini-2.5-flash'):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def sentiment_analyzer(product_id, collection, model, product_stats_data, llm_model_name='gemini-2.5-flash'):
    """
    Analyzes product reviews using the Gemini API to extract insights and return a structured JSON object.
    """
    stats = product_stats_data.get(product_id)

    if not stats:
        return {"error": f"Product with ID '{product_id}' not found."}
    
    try:
        reviews = collection.get(where={"product_id": product_id}, include=['documents', 'metadatas'])
        review_texts = reviews.get('documents', [])
        metadatas = reviews.get('metadatas', [])
        # Pair each review text with its metadata
        review_pairs = [
            (doc, meta) for doc, meta in zip(review_texts, metadatas)
            if doc and meta and 'review_creation_date' in meta
        ]
        if not review_pairs:
            return {
                "sentiment": "Indeterminado",
                "summary": "Não há avaliações suficientes para análise.",
                "positive_points": [],
                "negative_points": [],
                "top_reviews": []
            }
        
        # Sort by review_creation_date (most recent first)
        review_pairs.sort(key=lambda x: x[1]['review_creation_date'], reverse=True)
        # Select the most recent 100 reviews
        reviews_to_analyze = [doc for doc, meta in review_pairs[:100]]
        top_reviews = [doc for doc, meta in review_pairs[:3]]

    except Exception as e:
        return {"error": f"Failed to fetch data from ChromaDB: {e}"}
    reviews_formatted = "\n- ".join(reviews_to_analyze)
    prompt = f"""
    Você é um especialista em análise de marketing e reviews de produtos. Sua tarefa é analisar um conjunto
    de avaliações de clientes para um produto escolhido.
        
    A nota média das avaliações deste produto é {stats.get('average_score', 0.0)} (de 5). 
    Considere tanto essa nota média quanto o conteúdo textual das avaliações para classificar o sentimento geral, 
    e retorne um objeto JSON estruturado da seguinte maneira:

    1.  Sentimento Geral ('sentiment'): Com base em todas as avaliações, classifique o sentimento geral como "Positivo", "Negativo" ou "Neutro".
    2.  Resumo ('summary'): Escreva um resumo conciso de uma ou duas frases que capture a opinião dos clientes.
    3.  Pontos Positivos ('positive_points'): Extraia uma lista (array) de até 5 pontos positivos chave mencionados. Se nenhum for encontrado, retorne uma lista vazia '[]'.
    4.  Pontos Negativos ('negative_points'): Extraia uma lista (array) de até 3 pontos negativos chave. Se nenhum for encontrado, retorne uma lista vazia '[]'.

    
    - A sua resposta deve ser APENAS o objeto JSON.
    - Não inclua explicações, introduções, ou a palavra "json" no início.
    - Certifique-se de que o JSON esteja sintaticamente correto.

    **Avaliações para Análise:**
    - {reviews_formatted}

    **JSON DE SAÍDA ESPERADO:**
    """
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        analysis_result = json.loads(response.text)
    except Exception as e:
        return {"error": f"Failed to process the API response: {e}"}
    final_response = {
        "review_count": stats.get('review_count', 0),
        "average_score": stats.get('average_score', 0.0),
        **analysis_result,
        "top_reviews": top_reviews
    }
    return final_response