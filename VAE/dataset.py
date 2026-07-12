import torch
from torch.utils.data import Dataset
from scipy import sparse
import numpy as np

class Dataset_vae(Dataset):
    """
    PyTorch Dataset for VAE model.

    Important Args:
        embeddings : Dictionary mapping perturbation_name to embedding vector (numpy array)
    """
    def __init__(self, adata, target_col, control_name, embeddings, **kwargs):
        self.adata = adata
        self.pert = adata.obs[target_col]
        self.embeddings = embeddings
        self.x = adata.X
        ce_dim = next(iter(self.embeddings.values())).shape[0]
        self.e = np.stack([
            np.zeros(ce_dim) if p == control_name else self.embeddings[p]
            for p in self.pert
        ])

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx]
        if sparse.issparse(x):
            x = x.toarray().squeeze()
        x = torch.tensor(x, dtype=torch.float32)
        e = torch.tensor(self.e[idx], dtype=torch.float32)
        return x, e
    
