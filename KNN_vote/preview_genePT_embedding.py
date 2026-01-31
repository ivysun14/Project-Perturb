'''
preview_genePT_embedding.py
===========================

This scripts preview pre-computed GenePT embeddings. There are two forms of embeddings:
1. Embeddings from gene NCBI summary
2. Embeddings from gene NCBI summary and UniProt summary if available
'''

import zipfile
import pickle
import numpy as np

zip_path = 'data/GenePT_emebdding_v2.zip'

# List all files in the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Files in archive:")
    for name in zip_ref.namelist():
        print(f"  {name}")
    
    # Read two versions of GenePT embeddings
    pickle_path1 = 'GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'
    pickle_path2 = 'GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.'
    
    with zip_ref.open(pickle_path1) as f:
        gene_embeddings_NCBI = pickle.load(f)

    with zip_ref.open(pickle_path2) as f:
        gene_embeddings_NCBI_UniProt = pickle.load(f)

# Preview the structure
print("\nPreviewing GenePT embeddings from NCBI summary:")
print(f"Type: {type(gene_embeddings_NCBI)}")
print(f"Number of genes: {len(gene_embeddings_NCBI)}")
print(f"First 5 genes: {list(gene_embeddings_NCBI.keys())[:5]}")
first_gene_NCBI = list(gene_embeddings_NCBI.keys())[0]
print(f"Example gene: {first_gene_NCBI}")
print(f"Embedding data type: {type(gene_embeddings_NCBI[first_gene_NCBI])}")
print(f"Embedding shape: {len(gene_embeddings_NCBI[first_gene_NCBI])}")
print(f"First 10 values: {gene_embeddings_NCBI[first_gene_NCBI][:10]}")

print("\nPreviewing GenePT embeddings from NCBI and UniProt summary if available:")
print(f"Type: {type(gene_embeddings_NCBI_UniProt)}")
print(f"Number of genes: {len(gene_embeddings_NCBI_UniProt)}")
print(f"First 5 genes: {list(gene_embeddings_NCBI_UniProt.keys())[:5]}")
first_gene_NCBI_UniProt = list(gene_embeddings_NCBI_UniProt.keys())[0]
print(f"Example gene: {first_gene_NCBI_UniProt}")
print(f"Embedding data type: {type(gene_embeddings_NCBI_UniProt[first_gene_NCBI_UniProt])}")
print(f"Embedding shape: {len(gene_embeddings_NCBI_UniProt[first_gene_NCBI_UniProt])}")
print(f"First 10 values: {gene_embeddings_NCBI_UniProt[first_gene_NCBI_UniProt][:10]}")