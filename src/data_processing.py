import pandas as pd
import kagglehub
import os
import json

def download_and_load_data():
    """Download dataset from KaggleHub and load CSVs as DataFrames."""
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    df_reviews = pd.read_csv(f"{path}/olist_order_reviews_dataset.csv")
    df_orders = pd.read_csv(f"{path}/olist_orders_dataset.csv")
    df_order_items = pd.read_csv(f"{path}/olist_order_items_dataset.csv")
    return df_reviews, df_orders, df_order_items

def merge_data(df_reviews, df_orders, df_order_items):
    """Merge reviews, orders, and order items into a single DataFrame."""
    df_merged = pd.merge(df_reviews, df_orders, on='order_id')
    df_complete = pd.merge(df_merged, df_order_items, on='order_id')
    return df_complete

def compute_product_stats(df_complete, output_path="data/product_stats.json"):
    """Compute product statistics and save as JSON."""
    df_complete['review_score'] = pd.to_numeric(df_complete['review_score'], errors='coerce')
    product_stats = df_complete.groupby('product_id').agg(
        average_score=('review_score', 'mean'),
        review_count=('review_id', 'nunique')
    ).reset_index()
    product_stats['average_score'] = product_stats['average_score'].round(2)
    stats_dict = product_stats.set_index('product_id').to_dict('index')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=4)
    return stats_dict

def filter_reviews(df_complete, min_reviews=10):
    """Filter products with at least min_reviews and sort by average score ascending."""
    product_stats = df_complete.groupby('product_id').agg(
        average_score=('review_score', 'mean'),
        review_count=('review_id', 'nunique')
    ).reset_index()
    filtered = product_stats[product_stats['review_count'] >= min_reviews].sort_values('average_score', ascending=True)
    return filtered

def prepare_review_texts(df_complete):
    """Prepare DataFrame with combined review text for embedding."""
    cols_interest = ['product_id', 'review_score', 'review_comment_title', 'review_comment_message', 'review_creation_date']
    df_proc = df_complete[cols_interest].copy()
    df_proc['review_comment_title'] = df_proc['review_comment_title'].fillna('')
    df_proc['review_comment_message'] = df_proc['review_comment_message'].fillna('')
    def combine_review_text(row):
        title = row['review_comment_title'].strip()
        message = row['review_comment_message'].strip()
        if title and message:
            return title + '. ' + message
        elif title:
            return title
        elif message:
            return message
        else:
            return ''
    df_proc['full_review_text'] = df_proc.apply(combine_review_text, axis=1)
    df_with_text = df_proc[df_proc['full_review_text'] != ''].copy()
    return df_with_text
