## Weighted KNN Model for 3-Class Gene Perturbation Classification

### Overview
Weighted K-Nearest Neighbors (KNN) classifier to predict gene perturbation effects as a 3-class classification problem:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Non-DE | Not differentially expressed |
| 1 | Up-regulated | DE with positive log2 fold change |
| 2 | Down-regulated | DE with negative log2 fold change |

**Input**: Perturbation embeddings (3072-dimensional vectors derived from gene descriptions) and differential expression labels.

**Output**: Predictions with class probabilities for each (perturbation, gene) pair.

### Scripts

#### 1. `knn_classifier.py`
Main classifier with cross-validation and evaluation.
```bash
# For running CV for k selection
python -u knn_classifier.py \
    --de_doc_file "../baseline/files/DE_DOC_rpe1_splitted.csv" \
    --embedding_file "data/GenePT_emebdding_v2.zip" \
    --embedding_file2 "data/gene_embeddings.pkl" \
    --dataset_name "rpe1" \
    --k_values "1,3,5,7,9" \
    --n_folds 5 \
    --weights "distance" \
    --random_seed 305 \
    --output_dir "results"

# For skipping CV with a pre-defined k
python -u knn_classifier.py \
    --de_doc_file "../baseline/files/DE_DOC_rpe1_splitted.csv" \
    --embedding_file "data/GenePT_emebdding_v2.zip" \
    --embedding_file2 "data/gene_embeddings.pkl" \
    --dataset_name "rpe1" \
    --skip_cv \
    --k 3 \
    --weights "distance" \
    --random_seed 305 \
    --output_dir "results" \
```

#### 2. `fetch_gene_descriptions.py`
NCBI/UniProt gene description fetcher.
```bash
python fetch_gene_descriptions.py > logs/out.txt
```

#### 3. `generate_embeddings.py`
OpenAI embedding generation for missing genes.
- Uses OpenAI's `text-embedding-3-large` model (3072 dimensions)
- For genes without descriptions: embeds gene name only
```bash
python generate_embeddings.py > logs/out.txt
```

### Distance and Weighting

Distance Metric: **Euclidean distance** between perturbation embeddings in 3072-dimensional space

Weighting: **Inverse distance weighting** such that closer neighbors contribute more to the prediction:

For each test sample:
1. Find K nearest training perturbations
2. Weight each neighbor's vote by inverse distance
3. Accumulate weighted votes per class
4. Predict class with maximum total vote

Cross-validation selects optimal K (default candidates: 1, 3, 5, 7, 9) based on balanced accuracy.