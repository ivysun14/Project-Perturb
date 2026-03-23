# KNN Vote

Weighted k-nearest neighbours (k-NN) classifier for 3-class differential expression (DE) prediction (up / down / non-DE) in perturb-seq experiments.

## Algorithm

Each perturbed gene (perturbation) is represented as an embedding vector. To predict labels for a test perturbation, the classifier finds its K nearest training perturbations and aggregates their known DE labels for each target gene via weighted voting.

**Finding neighbours:** cosine similarity between perturbation embeddings. Three embedding types are supported:
- **GenePT** — text embeddings of gene descriptions fetched from NCBI/UniProt, encoded with OpenAI's `text-embedding-3-large` (3072-d).
- **ESM2** — protein sequence embeddings.
- **Co-pathway** — no embedding vectors; instead, neighbours are pre-ranked by Jaccard similarity of shared pathway memberships (MSigDB/KEGG/Reactome).

**Voting:** each neighbour casts a vote for each class (0/1/2) weighted by its cosine similarity to the query (or Jaccard similarity for co-pathway). The class with the highest total weight wins. Ties are broken at random. If a neighbour lacks a label for the target gene, the search expands to the next neighbour; genes with no coverage anywhere default to non-DE.

## Scripts

### Classify — custom DE/DOC file

```bash
# With cross-validation for K selection
python -u knn_classifier.py \
    --de_doc_file ../data/DE_DOC_rpe1_splitted.csv \
    --embedding_file data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2 data/additional_genePT_embeddings.pkl \
    --dataset_name rpe1 \
    --k_values "1,3,5,7,9" \
    --n_folds 5 \
    --weights distance \
    --random_seed 305 \
    --output_dir results

# Skip CV, use a fixed K
python -u knn_classifier.py \
    --de_doc_file ../data/DE_DOC_rpe1_splitted.csv \
    --embedding_file data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2 data/additional_genePT_embeddings.pkl \
    --dataset_name rpe1 \
    --skip_cv --k 3 \
    --weights distance \
    --random_seed 305 \
    --output_dir results

# For ESM2 switch out the embedding file above

# Co-pathway neighbours instead of an embedding file
python -u knn_classifier.py \
    --de_doc_file ../data/DE_DOC_rpe1_splitted.csv \
    --copath_file data/gene_copath_neighbors.json \
    --dataset_name rpe1_copath \
    --skip_cv --k 10 \
    --output_dir results
```

### Classify — PerturbQA pre-split files

```bash
python -u knn_classifier_perturbQA.py \
    --de_file ../data/perturbqa_datasets/k562-de.csv \
    --doc_file ../data/perturbqa_datasets/k562-dir.csv \
    --embedding_file data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2 data/additional_genePT_embeddings.pkl \
    --dataset_name k562 \
    --k_values "1,3,5,7,9" \
    --n_folds 5 \
    --weights distance \
    --random_seed 305 \
    --output_dir results
```

### Cross-cell-line transfer

```bash
python -u knn_classifier_transfer.py \
    --train_de_doc_file ../data/DE_DOC_K562_Replogle_splitted.csv \
                        ../data/DE_DOC_rpe1_splitted.csv \
                        ../data/DE_DOC_hepg2_splitted.csv \
    --test_de_doc_file  ../data/DE_DOC_jurkat_splitted.csv \
    --embedding_file    data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2   data/additional_genePT_embeddings.pkl \
    --train_dataset_name K562_Replogle rpe1 hepg2 \
    --test_dataset_name  jurkat_10NN \
    --k 10 \
    --weights distance \
    --random_seed 305 \
    --output_dir results
```

### Data preparation utilities

```bash
# Fetch gene descriptions from NCBI/UniProt
python fetch_gene_descriptions.py > logs/out.txt

# Generate OpenAI embeddings for genes
python generate_embeddings.py > logs/out.txt

# Pre-compute co-pathway neighbours (Jaccard similarity)
python precompute_pathways.py

# Check embedding/co-path coverage across datasets
python check_coverage.py
```

## Utilities

- **`analysis_utility.py`** — `load_embedding_from_pkl()`, `load_copath_neighbors()`, `build_embedding_matrix()`, `find_k_nearest_neighbors()`, `run_bootstrap()`
