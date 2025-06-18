import chromadb
import os
from tqdm import tqdm

def get_chromadb_client(db_path):
    """Create or load a persistent ChromaDB client."""
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)

def get_or_create_collection(client, collection_name):
    """Get or create a ChromaDB collection by name."""
    return client.get_or_create_collection(name=collection_name)

def add_documents_in_batches(collection, ids, embeddings, documents, metadatas, batch_size=4096):
    """Add documents to ChromaDB collection in batches."""
    for i in tqdm(range(0, len(ids), batch_size), desc="Adding batches to ChromaDB"):
        end_index = min(i + batch_size, len(ids))
        batch_ids = ids[i:end_index]
        batch_embeddings = embeddings[i:end_index]
        batch_documents = documents[i:end_index]
        batch_metadatas = metadatas[i:end_index]
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
