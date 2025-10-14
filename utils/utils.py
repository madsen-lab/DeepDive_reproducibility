import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse
import scipy
from wpca import WPCA
from anndata import AnnData
import torch
import scanpy as sc

def reads_to_fragments(
    adata,
    layer = None,
    key_added = None,
    copy = False,
):
    if copy:
        adata = adata.copy()

    if layer:
        data = np.ceil(adata.layers[layer].data / 2)
    else:
        data = np.ceil(adata.X.data / 2)

    if key_added:
        adata.layers[key_added] = adata.X.copy()
        adata.layers[key_added].data = data
    elif layer and key_added is None:
        adata.layers[layer].data = data
    elif layer is None and key_added is None:
        adata.X.data = data
    if copy:
        return adata.copy()
    
def plot_marker_heatmap(adata, markers, figsize=(2, 2)):
    df = adata.to_df()
    df['cell_type'] = adata.obs['cell_type']
    df = df.groupby('cell_type').mean()
    df = df.loc[markers.keys()]
    dfs = [df.loc[:,markers[i]] for i in markers.keys()]
    df = pd.concat(dfs, axis = 1)
    df_zscore = (df - df.mean())/df.std()
    fig, ax = plt.subplots(figsize = figsize)
    sns.heatmap(df_zscore, 
                    cmap = 'Blues', xticklabels=False, ax = ax)
    ax.set_xlabel("Peaks")
    ax.set_ylabel("")
    plt.tight_layout()

def subet_markers(markers, n):
    return {x:markers[x][:n].tolist() for x in markers.keys()}
    

def set_dropout(model, drop_rate=0.0):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)
        


import scipy.sparse
def get_dense_matrix(adata, max_features=None):
    M = adata.X
    if scipy.sparse.issparse(M):
        M = M.toarray()
    if max_features is not None:
        tot = np.sum(M, axis=0)
        top_idx = np.argsort(tot)[-max_features:]
        M = M[:, top_idx]
    M = M.astype(float)
    return M

def w_dimred(emb, weights = None):
    kwds = {'weights': np.array([weights]*emb.shape[1]).T}
    X = WPCA(n_components=2).fit_transform(emb, **kwds)
    return X



def plot_embedding_joint(emb, weights, labels_name = 'cell_type', label_cap=10):
    labels_all = emb.index.tolist()
    label_from = emb['group'].values
    weights_all = [weights[x] for x in label_from]
    weights_all.append(-1)
    emb.drop('group', axis = 1, inplace = True)
    emb.loc['center'] = 0
    
    emb_decomp = w_dimred(emb.values, weights = weights_all)
    center = emb_decomp[emb_decomp.shape[0] - 1]
    emb_decomp = emb_decomp[:-1,:]
    
    df = pd.DataFrame(emb_decomp, columns=["dim1", "dim2"])
    df['group'] = label_from
    labels = list(df['group'].unique())

    x_axis = "dim1"
    y_axis = "dim2"
    
    circle_size=40
    circe_transparency=1.0
    line_transparency=0.8
    line_width=1.0
    fontsize=9
    fig_width=4
    fig_height=4
    file_name=None
    file_format=None
    width_ratios=[7, 1]
    bbox=(1.3, 0.7)
    fontsize = 10
    title = labels_name
    n_colors = len(labels)
    palette = sns.color_palette('Set1')
    col_dict = dict(zip(labels, palette[:n_colors]))
    fig, axs = plt.subplots(1,len(labels), figsize=(len(labels)*2.3,2), sharex = True, sharey = True)
    for idx, i in enumerate(labels):
        df_sub = df[df['group'] == i]    
        sns.scatterplot(
                x="dim1",
                y="dim2",
                c = palette[idx],
                alpha=circe_transparency,
                edgecolor='k',
                s=circle_size,
                data=df_sub,
                ax=axs[idx],
                legend = False, 
                
            )      
        try:
            ax.legend_.remove()
        except:
            pass

        for j in range(len(df_sub)):
            axs[idx].plot(
                [center[0], df_sub.values[j, 0]],
                [center[1], df_sub.values[j, 1]],
                alpha=line_transparency,
                linewidth=line_width,
                c='k',
            )
        texts = []
        labels = [labels_all[xdx] for xdx in df_sub.index.tolist()]
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        if len(unique_labels) < label_cap:
            for label in unique_labels:
                idx_label = np.where(labels == label)[0]
                texts.append(
                    axs[idx].text(
                        np.mean(df_sub.values[idx_label, 0]),
                        np.mean(df_sub.values[idx_label, 1]),
                        label,
                    )
                )
        axs[idx].set_title(i)
    return fig

def mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match, got {a.shape} vs {b.shape}")
    return np.mean(np.square(a - b))

def get_colormap_colors(colormap_name, n_colors):
    cmap = plt.cm.get_cmap(colormap_name, n_colors)
    return [cmap(i) for i in range(n_colors)]

def preprocess(adata, discrete_covariates, continuous_covariates, frac = 0.05):
    adata.obs_names_make_unique()
    adata = reads_to_fragments(adata, copy = True)
    min_cells = int(adata.shape[0] * frac)
    sc.pp.filter_genes(adata, min_cells=min_cells, inplace = True)
    sc.pp.filter_cells(adata, min_genes=5)
    float_from_str = lambda x: float(x.replace(',', '.')) if type(x) == str else float(x)
    adata.obs[continuous_covariates] = adata.obs[continuous_covariates].apply(lambda x:[float_from_str(i) for i in x])
    adata.obs = adata.obs[discrete_covariates+continuous_covariates]
    return adata

def sample_cells_per_donor(adata, donor_key="donor", n_cells=2000, random_state=0):

    np.random.seed(random_state)
    sampled_indices = []
    
    for donor, idxs in adata.obs.groupby(donor_key).indices.items():
        if len(idxs) > n_cells:
            chosen = np.random.choice(idxs, n_cells, replace=False)
        else:
            chosen = idxs 
        sampled_indices.extend(chosen)
    
    return adata[sampled_indices].copy()

def tfidf_transform(adata: AnnData, layer: str = None, log: bool = False) -> None:

    from scipy.sparse import csr_matrix, issparse

    X = adata.layers[layer] if layer is not None else adata.X
    if not issparse(X):
        X = csr_matrix(X)

    # Term frequency (TF)
    tf = X.multiply(1 / (np.asarray(X.sum(axis=1)).ravel()[:, None]))

    # Inverse document frequency (IDF)
    n_cells = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(n_cells / (1 + df))

    # Apply TF-IDF
    tfidf = tf.multiply(idf)

    if log:
        tfidf.data = np.log1p(tfidf.data)

    # Store result in adata
    adata.X = csr_matrix(tfidf)
    adata.uns["tfidf_applied"] = True
