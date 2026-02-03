"""
generate_embeddings.py
=======================

Generate GenePT embeddings using OpenAI API.

Prerequisites:
1. Run fetch_gene_descriptions.py first to create data/all_gene_descriptions.json
2. Set OPENAI_API_KEY environment variable

Input: data/all_gene_descriptions.json
Output: data/gene_embeddings.pkl

Following: https://github.com/yiqunchen/GenePT/blob/main/gene_embeddings_examples.ipynb
"""

import os
import json
import pickle
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

# Configuration
GPT_DIM = 3072  # Dimension of text-embedding-3-large
EMBEDDING_MODEL = "text-embedding-3-large"
INPUT_FILE = "data/all_gene_descriptions.json"
OUTPUT_FILE = "data/gene_embeddings.pkl"

# Gene list
GENE_LIST = ["AC118549.1", "ALG1L", "C12orf45", "C1orf109", "C7orf26",
            "C9orf16", "CCDC130", "CCDC144NL", "DDN", "FRG2", "MBTPS1",
            "PHB", "POLR2B", "SPATA5L1", "TDGF1", "WDR61", "WDR92", "ZNF720",
            "FAM102B", "INTS2", "SPATA5"]

def get_gpt_embedding(client: OpenAI, text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Get embedding from OpenAI API.

    Args:
        client: OpenAI client (need API key)
        text: Input text to embed
        model: OpenAI embedding model to use

    Returns:
        numpy array of shape (GPT_DIM,)
    """
    
    text = text.replace("\n", " ")
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


if __name__ == "__main__":

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)
    print(f"Using OpenAI embeddings model: {EMBEDDING_MODEL}")

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        raise SystemExit(1)

    with open(INPUT_FILE, "r") as f:
        gene_descriptions = json.load(f)
    print(f"Loaded {len(gene_descriptions)} gene descriptions from {INPUT_FILE}")

    gene_embeddings = {}
    genes_with_embeddings = []
    genes_with_zero_vectors = []

    for i, (gene_name, description) in enumerate(sorted(gene_descriptions.items())):
        if description and description.strip():
            print(f"[{i+1}/{len(gene_descriptions)}] Embedding: {gene_name}")
            try:
                emb = get_gpt_embedding(client, description)
                gene_embeddings[gene_name] = emb
                genes_with_embeddings.append(gene_name)
            except Exception as e:
                print(f"  ERROR: {e}")
                gene_embeddings[gene_name] = np.zeros(GPT_DIM, dtype=np.float32)
                genes_with_zero_vectors.append(gene_name)
        else:
            print(f"[{i+1}/{len(gene_descriptions)}] Embedding gene name only: {gene_name}")
            try:
                emb = get_gpt_embedding(client, gene_name)
                gene_embeddings[gene_name] = emb
                genes_with_embeddings.append(gene_name)
            except Exception as e:
                print(f"  ERROR: {e}")
                gene_embeddings[gene_name] = np.zeros(GPT_DIM, dtype=np.float32)
                genes_with_zero_vectors.append(gene_name)
        print(f"Generated embedding of shape: {gene_embeddings[gene_name].shape}")
        time.sleep(0.50)  # wait 0.50 seconds before next request

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(gene_embeddings, f)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Total embeddings generated: {len(gene_embeddings)}")
    print(f"Genes with real embeddings: {len(genes_with_embeddings)}")
    print(f"Genes with zero vectors: {len(genes_with_zero_vectors)}")
    if genes_with_zero_vectors:
        print(f"\nGenes with zero vectors: {genes_with_zero_vectors}")