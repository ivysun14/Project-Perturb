"""
Precompute nearest neighbors for all genes in embedding space.

Creates JSON files with top-K nearest neighbors for fast lookup during evolution.
Output: {gene_name: [neighbor1, neighbor2, ..., neighborK]}

Optimizations:
- Skip normalization (embeddings already unit norm)
- Use argpartition instead of full sort (O(n) vs O(n log n))
- Interruption-safe: saves progress incrementally
"""

import os
import pickle
import json
import numpy as np
from tqdm import tqdm

EMBEDDING_DIR = os.path.expanduser("~/Downloads/data_embedding")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_K = 20


def load_embeddings(name: str, path: str):
    """Load embeddings from pickle file."""
    print(f"Loading {name} from {path}...")
    with open(path, 'rb') as f:
        emb_dict = pickle.load(f)
    print(f"  Loaded {len(emb_dict)} genes")
    return emb_dict


def compute_nearest_neighbors_fast(emb_dict: dict, output_path: str, top_k: int = 20,
                                    skip_normalize: bool = True, save_interval: int = 5000):
    """
    Compute top-K nearest neighbors using fast argpartition.
    Saves incrementally for interruption safety.
    """
    genes = list(emb_dict.keys())
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    print(f"Building embedding matrix for {n_genes} genes...")

    # Build matrix
    emb_list = []
    for g in genes:
        emb = emb_dict[g]
        if isinstance(emb, (list, tuple)):
            emb = np.array(emb)
        emb_list.append(emb.flatten())

    emb_matrix = np.vstack(emb_list).astype(np.float32)  # float32 is faster
    print(f"  Matrix shape: {emb_matrix.shape}")

    if not skip_normalize:
        print("Normalizing embeddings...")
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_matrix = emb_matrix / norms
    else:
        print("Skipping normalization (already unit norm)")

    # Load existing progress if any
    neighbors = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                neighbors = json.load(f)
            print(f"  Resuming from {len(neighbors)} already computed genes")
        except:
            neighbors = {}

    # Compute neighbors in batches
    print(f"Computing top-{top_k} neighbors (argpartition)...")
    batch_size = 2000  # Larger batches for efficiency

    genes_to_compute = [g for g in genes if g not in neighbors]
    print(f"  {len(genes_to_compute)} genes remaining")

    for batch_start in tqdm(range(0, len(genes_to_compute), batch_size)):
        batch_end = min(batch_start + batch_size, len(genes_to_compute))
        batch_genes = genes_to_compute[batch_start:batch_end]
        batch_indices = [gene_to_idx[g] for g in batch_genes]
        batch_emb = emb_matrix[batch_indices]

        # Dot product for cosine similarity (already normalized)
        sim_matrix = batch_emb @ emb_matrix.T  # (batch, n_genes)

        for i, gene in enumerate(batch_genes):
            gene_idx = batch_indices[i]
            sim_scores = sim_matrix[i]

            # Set self-similarity to -inf to exclude
            sim_scores[gene_idx] = -np.inf

            # argpartition is O(n) - only partially sorts to find top-k
            # Get indices of top_k largest (they won't be sorted)
            top_indices = np.argpartition(sim_scores, -top_k)[-top_k:]

            # Sort just those top_k by their actual scores
            top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]

            neighbors[gene] = [genes[idx] for idx in top_indices]

        # Save progress periodically
        if (batch_start + batch_size) % save_interval < batch_size:
            print(f"\n  Saving checkpoint ({len(neighbors)} genes)...")
            with open(output_path, 'w') as f:
                json.dump(neighbors, f)

    # Final save
    print(f"Saving final results ({len(neighbors)} genes)...")
    with open(output_path, 'w') as f:
        json.dump(neighbors, f)

    return neighbors


def main():
    # ESM2 embeddings (skip since already done)
    esm2_path = os.path.join(EMBEDDING_DIR, "ESM2_protein_embedding.pkl")
    esm2_output = os.path.join(OUTPUT_DIR, "gene_neighbors_esm2.json")

    if os.path.exists(esm2_output):
        with open(esm2_output) as f:
            n = len(json.load(f))
        print(f"ESM2 neighbors already computed ({n} genes), skipping...")
    elif os.path.exists(esm2_path):
        esm2_emb = load_embeddings("ESM2", esm2_path)
        compute_nearest_neighbors_fast(esm2_emb, esm2_output, TOP_K, skip_normalize=True)

    # GenePT embeddings
    genept_path = os.path.join(EMBEDDING_DIR, "GPT_model_3_gene_and_protein.pickle")
    genept_output = os.path.join(OUTPUT_DIR, "gene_neighbors_genept.json")

    if os.path.exists(genept_path):
        genept_emb = load_embeddings("GenePT/GPT", genept_path)
        compute_nearest_neighbors_fast(genept_emb, genept_output, TOP_K, skip_normalize=True)

    print("\nDone! Created:")
    print(f"  - gene_neighbors_esm2.json (top {TOP_K} neighbors per gene)")
    print(f"  - gene_neighbors_genept.json (top {TOP_K} neighbors per gene)")


if __name__ == "__main__":
    main()
