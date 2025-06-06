import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from pinecone import Pinecone, ServerlessSpec
import os
import nltk
from sentence_transformers import SentenceTransformer

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Step 1: Load and process CSV data
def process_csv_data(data_path):
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure DESCRIPTION column exists
    if 'DESCRIPTION' not in df.columns:
        print("Error: CSV file must contain a 'DESCRIPTION' column")
        return None, None, None
    
    # Clean and prepare text data
    descriptions = []
    for desc in df['DESCRIPTION']:
        if not pd.isna(desc):
            descriptions.append(str(desc).lower())
        else:
            descriptions.append("")
    
    # Create a unique ID for each description
    ids = [f"item_{i}" for i in range(len(descriptions))]
    
    # Add any metadata you want to store
    metadata = []
    for i, row in df.iterrows():
        meta_dict = {"id": f"item_{i}"}
        # Add other columns as metadata if they exist
        for col in df.columns:
            if col != 'DESCRIPTION' and not pd.isna(row[col]):
                meta_dict[col] = str(row[col])
        metadata.append(meta_dict)
    
    print(f"Processed {len(descriptions)} descriptions from CSV")
    return descriptions, ids, metadata

# Step 2: Generate embeddings from text (using BERT for 768-dimensions)
def generate_embeddings(descriptions, target_dim=768):
    print(f"Generating {target_dim}-dimensional embeddings using BERT")
    
    # Use a model that produces 768-dimensional embeddings
    model = SentenceTransformer('bert-base-uncased')  # 768 dimensions by default
    embeddings = model.encode(descriptions)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    if embeddings.shape[1] != target_dim:
        print(f"Warning: Embeddings dimension {embeddings.shape[1]} doesn't match target dimension {target_dim}")
        # Adjust dimensions if needed (can expand with zeros or truncate)
        if embeddings.shape[1] < target_dim:
            # Pad with zeros
            padded_embeddings = np.zeros((len(embeddings), target_dim))
            padded_embeddings[:, :embeddings.shape[1]] = embeddings
            embeddings = padded_embeddings
        else:
            # Truncate
            embeddings = embeddings[:, :target_dim]
        print(f"Adjusted embeddings to shape: {embeddings.shape}")
    
    return embeddings

# Step 3: Initialize Pinecone client
def init_pinecone(api_key, environment="us-east-1"):
    try:
        pc = Pinecone(api_key=api_key)
        print(f"Pinecone client initialized with environment '{environment}'")
        return pc
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None

# Step 4: Get existing Pinecone index
def get_pinecone_index(pc, index_name):
    try:
        existing_indexes = [i.name for i in pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Error: Index '{index_name}' does not exist. Available indexes: {existing_indexes}")
            return None, None
        else:
            print(f"Using existing index '{index_name}'")
            index_info = pc.describe_index(index_name)
            index_dimension = index_info.dimension
            print(f"Index dimension: {index_dimension}")
            return pc.Index(index_name), index_dimension
    except Exception as e:
        print(f"Error getting Pinecone index: {e}")
        return None, None

# Step 5: Upload embeddings to Pinecone
def upload_embeddings_to_pinecone(index, embeddings, ids, metadata, batch_size=100):
    total_vectors = len(embeddings)
    print(f"Uploading {total_vectors} vectors to Pinecone in batches of {batch_size}")
    
    for i in tqdm(range(0, total_vectors, batch_size)):
        batch_ids = []
        batch_vectors = []
        batch_metadata = []
        end_idx = min(i + batch_size, total_vectors)
        
        for j in range(i, end_idx):
            batch_ids.append(ids[j])
            batch_vectors.append(embeddings[j].tolist())
            batch_metadata.append(metadata[j])
        
        try:
            index.upsert(vectors=zip(batch_ids, batch_vectors, batch_metadata))
        except Exception as e:
            print(f"Error uploading batch {i // batch_size}: {e}")
            print(f"Error details: {e}")
            return False
    
    print(f"Successfully uploaded {total_vectors} vectors to Pinecone")
    return True

# Main function to orchestrate everything
def csv_to_pinecone():
    # Configuration
    PINECONE_API_KEY = "pcsk_4g6RKv_JhPB9LtE1QSjcf8jD246iLRWu6CwZrqCNKHaSVLb4vpv3TxTjdzBspeXc81wpDu"
    INDEX_NAME = "stylingag"  # Use your existing index
    DATA_PATH = "/Users/shriya/Documents/GitHub/logo_detect/dsmlprojects/pc+ELmo/vasavi2.csv"

    # Process CSV data
    descriptions, ids, metadata = process_csv_data(DATA_PATH)
    if descriptions is None:
        print("Failed to process CSV data. Exiting.")
        return

    # Initialize Pinecone
    pc = init_pinecone(PINECONE_API_KEY)
    if pc is None:
        print("Failed to initialize Pinecone. Exiting.")
        return

    # Get existing Pinecone index and its dimension
    index, index_dimension = get_pinecone_index(pc, INDEX_NAME)
    if index is None:
        print("Failed to get Pinecone index. Exiting.")
        return

    # Generate embeddings with the correct dimension
    embeddings = generate_embeddings(descriptions, target_dim=index_dimension)
    
    # Upload embeddings to Pinecone
    success = upload_embeddings_to_pinecone(index, embeddings, ids, metadata)
    if not success:
        print("Failed to upload embeddings to Pinecone. Exiting.")
        return

    print(f"\nData from {DATA_PATH} successfully processed and saved to Pinecone index '{INDEX_NAME}'")

    # Example query
    print("\nExample query:")
    try:
        # For the example, we'll use the first item's embedding as a query
        query_response = index.query(
            vector=embeddings[0].tolist(),
            top_k=5,
            include_metadata=True
        )
        
        print("Query results:")
        for match in query_response['matches']:
            print(f"ID: {match['id']}, Score: {match['score']}")
            print(f"Metadata: {match['metadata']}")
            print("---")
    except Exception as e:
        print(f"Error running example query: {e}")

if __name__ == "__main__":
    csv_to_pinecone()