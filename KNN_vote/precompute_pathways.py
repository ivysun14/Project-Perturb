"""
precompute_pathways.py
======================
Precompute pathway/module-based gene groupings.

Creates JSON files with pathway co-membership for fast lookup during evolution.
Output:
  - gene_pathways.json: {gene: [pathway1, pathway2, ...]}
  - pathway_genes.json: {pathway: [gene1, gene2, ...]}
  - gene_copath_neighbors.json: {gene: [[neighbor, jaccard_sim], ...]}
    Each entry is a list of [neighbor_gene, jaccard_similarity] pairs sorted
    by descending Jaccard similarity (higher = more shared pathways).
"""

import os
import json
from collections import defaultdict
from tqdm import tqdm
import gseapy

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOP_K = 100  # Neighbor to find


def load_gene_sets():
    gene_sets = {}
    libraries = [
        'KEGG_2021_Human',
        'Reactome_2022',
        'GO_Biological_Process_2023',
        'WikiPathway_2023_Human',
        'MSigDB_Hallmark_2020',
    ]

    print("Loading gene sets from gseapy...")
    for lib in libraries:
        try:
            gs = gseapy.get_library(lib)
            gene_sets[lib] = gs
            print(f"  {lib}: {len(gs)} pathways")
        except Exception as e:
            print(f"  {lib}: Failed to load ({e})")

    return gene_sets


def build_gene_to_pathways(gene_sets: dict):
    """Build mapping from gene to list of pathways it belongs to."""
    gene_to_pathways = defaultdict(set)
    for lib_name, pathways in gene_sets.items():
        for pathway_name, genes in pathways.items():
            full_pathway = f"{lib_name}:{pathway_name}"
            for gene in genes:
                gene_to_pathways[gene].add(full_pathway)
    gene_to_pathways = {g: sorted(list(p)) for g, p in gene_to_pathways.items()}
    return gene_to_pathways


def build_pathway_to_genes(gene_sets: dict):
    """Build mapping from pathway to list of genes."""
    pathway_to_genes = {}
    for lib_name, pathways in gene_sets.items():
        for pathway_name, genes in pathways.items():
            full_pathway = f"{lib_name}:{pathway_name}"
            pathway_to_genes[full_pathway] = list(genes)

    return pathway_to_genes


def compute_copath_neighbors(gene_to_pathways: dict, top_k: int = 20):
    """
    For each gene, find top-K genes that share the most pathways.
    Uses Jaccard similarity on pathway sets.
    """
    genes = list(gene_to_pathways.keys())
    n_genes = len(genes)

    print(f"Computing co-pathway neighbors for {n_genes} genes...")
    pathway_sets = {g: set(gene_to_pathways[g]) for g in genes}
    neighbors = {}
    for gene in tqdm(genes):
        gene_pathways = pathway_sets[gene]
        if not gene_pathways:
            neighbors[gene] = []  # Gene has no pathway annotations; no neighbors possible
            continue

        # Compute Jaccard similarity with all other genes
        similarities = []
        for other in genes:
            if other == gene:
                continue
            other_pathways = pathway_sets[other]
            if not other_pathways:
                continue

            # Jaccard similarity
            intersection = len(gene_pathways & other_pathways)
            union = len(gene_pathways | other_pathways)
            sim = intersection / union if union > 0 else 0

            if sim > 0:  # Only keep non-zero similarities; genes sharing no pathways are excluded
                similarities.append((other, sim))

        # Sort by similarity (descending) and take top K.
        # NOTE: if a gene shares pathways with fewer than top_k other genes,
        # the resulting neighbor list will be shorter than top_k. Downstream
        # consumers should not assume exactly top_k neighbors exist per gene.
        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbors[gene] = [[g, sim] for g, sim in similarities[:top_k]]

    return neighbors


def main():
    gene_sets = load_gene_sets()
    if not gene_sets:
        print("No gene sets loaded!")
        return

    print("\nBuilding gene-to-pathway mapping...")
    gene_to_pathways = build_gene_to_pathways(gene_sets)
    print(f"  {len(gene_to_pathways)} genes with pathway annotations")

    print("\nBuilding pathway-to-gene mapping...")
    pathway_to_genes = build_pathway_to_genes(gene_sets)
    print(f"  {len(pathway_to_genes)} pathways")

    output_path = os.path.join(OUTPUT_DIR, "gene_pathways.json")
    print(f"\nSaving gene_pathways.json...")
    with open(output_path, 'w') as f:
        json.dump(gene_to_pathways, f)

    output_path = os.path.join(OUTPUT_DIR, "pathway_genes.json")
    print(f"Saving pathway_genes.json...")
    with open(output_path, 'w') as f:
        json.dump(pathway_to_genes, f)

    print("\nComputing co-pathway neighbors...")
    copath_neighbors = compute_copath_neighbors(gene_to_pathways, TOP_K)

    output_path = os.path.join(OUTPUT_DIR, "gene_copath_neighbors.json")
    print(f"Saving gene_copath_neighbors.json...")
    with open(output_path, 'w') as f:
        json.dump(copath_neighbors, f)

    print("\nDone!")


if __name__ == "__main__":
    main()