import numpy as np
import pandas as pd
import scanpy as sc

def sample_w_cramer(data, n_cells, cramers):
    groups = pd.Series([x for x in data.obs.Gender]) + '_' + pd.Series([x for x in data.obs.Disease]) 
    per_group_p = generate_dependency_matrix(2, 2, cramers) / pd.crosstab(data.obs.Gender, data.obs.Disease)
    props = [per_group_p.loc[x.split('_')[0], x.split('_')[1]] for x in groups]
    sampled = np.random.choice(data.obs.index, size = n_cells, p = props, replace = False)
    data_sampled = data[sampled]
    return data_sampled
    

def generate_dependency_matrix(n_groups1, n_groups2, entanglement_level):
    """
    Generate a dependency matrix representing the level of entanglement between two covariates.
    
    Parameters:
        n_groups1 (int): Number of groups for the first covariate.
        n_groups2 (int): Number of groups for the second covariate.
        entanglement_level (float): Level of entanglement (0.0 = no entanglement, 1.0 = perfect entanglement).
        
    Returns:
        np.array: Dependency matrix (n_groups1 x n_groups2).
    """
    # No entanglement (uniform probabilities)
    no_entanglement = np.full((n_groups1, n_groups2), 1 / (n_groups1 * n_groups2))
    
    # Perfect entanglement (identity matrix scaled to probabilities)
    perfect_entanglement = np.eye(min(n_groups1, n_groups2))
    perfect_entanglement = np.pad(
        perfect_entanglement,
        ((0, max(0, n_groups1 - n_groups2)), (0, max(0, n_groups2 - n_groups1))),
        'constant'
    ) / min(n_groups1, n_groups2)
    
    # Interpolate between no entanglement and perfect entanglement
    dependency_matrix = (1 - entanglement_level) * no_entanglement + entanglement_level * perfect_entanglement
    
    return dependency_matrix

def calc_de(data):
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)
    sc.tl.rank_genes_groups(data, "Gender", method="wilcoxon", groups = ['M'], reference = 'F')

    result = data.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    df_full_sex = pd.DataFrame(
            {
                f"{group}_{key[:1]}": result[key][group]
                for group in groups
                for key in ["names", "logfoldchanges", 'pvals_adj']
            }

        )

    sc.tl.rank_genes_groups(data, "Disease", method="wilcoxon", groups = ['T2D'], reference = 'Non')

    result = data.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    df_full_d = pd.DataFrame(
            {
                f"{group}_{key[:1]}": result[key][group]
                for group in groups
                for key in ["names", "logfoldchanges", 'pvals_adj']
            }

        )
    return df_full_sex, df_full_d