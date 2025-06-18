from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(texts, model_name='all-mpnet-base-v2', show_progress_bar=True):
    """
    Generate embeddings for a list of texts using a specified SentenceTransformer model.
    Args:
        texts (list): List of strings to embed.
        model_name (str): Name of the SentenceTransformer model.
        show_progress_bar (bool): Whether to show progress bar.
    Returns:
        np.ndarray: Embeddings array.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return embeddings
