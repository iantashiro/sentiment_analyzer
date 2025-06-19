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
        return {"error": f"Product with ID '{product_id}' not found in the statistics file."}
    
    try:
        reviews = collection.get(where={"product_id": product_id}, include=['documents', 'metadatas'])
        review_texts = reviews.get('documents', [])
        metadatas = reviews.get('metadatas', [])
        # Pair each review text with its metadata
        review_pairs = [
            (doc, meta) for doc, meta in zip(review_texts, metadatas)
            if doc and meta and 'review_creation_date' in meta and 'review_score' in meta
        ]
        if not review_pairs:
            return {
                "sentiment": "Indeterminado",
                "summary": "Não há avaliações suficientes para análise.",
                "positive_points": [],
                "negative_points": [],
                "top_reviews": []
            }
        # Sort by review_creation_date descending (most recent first)
        review_pairs.sort(key=lambda x: x[1]['review_creation_date'], reverse=True)
        # Select the most recent 100 reviews
        reviews_to_analyze = review_pairs[:100]
        top_reviews = [doc for doc, meta in review_pairs[:3]]

    except Exception as e:
        return {"error": f"Failed to fetch data from ChromaDB: {e}"}
    
    # Format reviews with score for the prompt
    reviews_formatted = "\n- ".join(
        f"[Nota: {meta['review_score']}] {doc}" for doc, meta in reviews_to_analyze
    )
    prompt = f"""
Você é um especialista em análise de marketing e reviews de produtos. Sua tarefa é analisar um conjunto
 de avaliações de clientes para um produto escolhido.

A nota média das avaliações deste produto é {stats.get('average_score', 0.0)} (de 5).
Cada avaliação está acompanhada de sua respectiva nota (de 1 a 5). Considere tanto a nota média quanto o 
texto e a nota individual de cada avaliação para classificar o sentimento geral de forma equilibrada, 
refletindo a percepção real dos clientes.

Se a maioria das avaliações e a nota média forem positivas, classifique como "Positivo", mesmo que existam 
algumas reclamações negativas. Use "Neutro" apenas se houver um equilíbrio claro entre avaliações positivas e 
negativas, e explique no resumo que as experiências dos clientes foram divergentes. Considere não apenas a 
quantidade, mas também a gravidade dos problemas relatados, por exemplo: problemas de qualidade ou defeitos 
são mais graves do que atrasos na entrega.

Retorne um objeto JSON estruturado da seguinte maneira:

1.  Sentimento Geral ('sentiment'): Classifique como "Positivo", "Negativo" ou "Neutro".
2.  Resumo ('summary'): Escreva um resumo conciso de uma ou duas frases que capture a opinião dos clientes e justifique a classificação do sentimento, especialmente em casos de equilíbrio.
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
