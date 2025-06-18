import os
from src import data_processing, embedding, database

def main():
    # Step 1: Data loading and cleaning
    df_reviews, df_orders, df_order_items = data_processing.download_and_load_data()
    df_complete = data_processing.merge_data(df_reviews, df_orders, df_order_items)

    # Step 2: Compute product statistics and save to data/product_stats.json
    stats_dict = data_processing.compute_product_stats(df_complete, output_path="data/product_stats.json")
    print(f"✅ Pre-computation complete. {len(stats_dict)} products saved to data/product_stats.json")
    print("Sample of the generated statistics:")
    print(dict(list(stats_dict.items())[:5]))

    # Step 3: Prepare review texts for embedding
    df_with_text = data_processing.prepare_review_texts(df_complete)
    print(f"Found {len(df_with_text)} reviews with text to process.")

    # Step 4: Generate embeddings
    review_texts_list = df_with_text['full_review_text'].tolist()
    embeddings = embedding.generate_embeddings(review_texts_list)
    print(f"Embeddings generated successfully! Shape: {embeddings.shape}")

    # Step 5: Store in ChromaDB
    db_path = "data/review_db"
    collection_name = "olist_reviews"
    client = database.get_chromadb_client(db_path)
    collection = database.get_or_create_collection(client, collection_name)

    ids = [f"review_{i}" for i in df_with_text.index]
    documents = df_with_text['full_review_text'].tolist()
    metadatas = df_with_text[['product_id', 'review_score', 'review_creation_date', 'review_comment_title', 'review_comment_message']].to_dict('records')

    database.add_documents_in_batches(collection, ids, embeddings, documents, metadatas)
    print(f"\n✅ Done! {collection.count()} reviews have been saved to ChromaDB in '{db_path}'.")

if __name__ == "__main__":
    main()
