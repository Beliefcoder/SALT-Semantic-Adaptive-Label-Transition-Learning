import os

import numpy as np
from scipy import sparse


def save_sparse(path, x):
    if x.ndim == 2:
        idx = np.where(x > 0)
        values = x[idx]
        np.savez(path, idx=idx, values=values, shape=x.shape)
    elif x.ndim == 3:
        idx = np.where(x > 0)
        values = x[idx]
        np.savez(path, idx=idx, values=values, shape=x.shape)
    elif x.ndim == 4:
        idx = np.where(x > 0)
        values = x[idx]
        np.savez(path, idx=idx, values=values, shape=x.shape)
    else:
        np.savez(path, data=x)


def load_sparse(path, return_sparse=True):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    shape = tuple(data['shape'])
    
    if len(shape) == 2:
        mat = np.zeros(shape, dtype=values.dtype)
        mat[tuple(idx)] = values
        if return_sparse:
            return sparse.csr_matrix(mat)
        return mat
    elif len(shape) == 3:
        mat = np.zeros(shape, dtype=values.dtype)
        mat[tuple(idx)] = values
        if return_sparse:
            return sparse.csr_matrix(mat.reshape(-1, shape[-1]))
        return mat
    elif len(shape) == 4:
        mat = np.zeros(shape, dtype=values.dtype)
        mat[tuple(idx)] = values
        if return_sparse:
            return sparse.csr_matrix(mat.reshape(-1, shape[-1]))
        return mat
    else:
        raise ValueError(f"Unsupported array dimension: {len(shape)}")


def save_data(path, code_x, visit_lens, codes_y, hf_y, divided, neighbors):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)
