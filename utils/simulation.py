import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import minmax_scale
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm


def simulate_scATAC_seq_data(n_cells, n_features, covariates, max_effect_sizes=[0.2, 0.2], sample_feature_p=[0.25, 0.25], dependency=None, p=0.5):
    """
    Simulate scATAC-seq data with covariates influencing fragment counts.
    
    Parameters:
        n_cells (int): Number of cells.
        n_features (int): Number of peaks/features.
        covariates (dict): Dictionary where keys are covariate names and values are lists of group labels.
        p (float): Probability of assigning a cell to the first group in each covariate (default is 0.5).
        max_effect_sizes (list): Maximum effect sizes for each covariate.
        sample_feature_p (list): Probabilities of selecting features for adjustment for each covariate.
        dependency (np.array or None): Optional. A joint probability matrix specifying the dependency between two covariates. 
                                        Shape should be (len(groups1), len(groups2)).

    Returns:
        dict: 
            - 'data': Fragment count matrix (cells x features).
            - 'effect_sizes': Adjustments applied to each covariate group.
            - 'covariate_assignments': Covariate group assignments for each cell.
            - 'pivot': Normalized pivot table showing overlap of covariate assignments.
    """
    # Step 1: Initialize mean accessibility for each peak
    base_accessibility = np.random.uniform(0.5, 2.0, n_features)
    
    # Step 2: Assign cells to covariate groups
    covariate_keys = list(covariates.keys())
    covariate_assignments = {}

    if dependency is not None:
        # Handle dependent covariates
        assert len(covariate_keys) == 2, "Dependency matrix can only be used for two covariates."
        groups1, groups2 = covariates[covariate_keys[0]], covariates[covariate_keys[1]]
        
        # Flatten the joint probability matrix and sample indices
        flat_indices = np.random.choice(len(dependency.flatten()), size=n_cells, p=dependency.flatten())
        
        # Map indices to group assignments for both covariates
        covariate_assignments[covariate_keys[0]] = [groups1[i // len(groups2)] for i in flat_indices]
        covariate_assignments[covariate_keys[1]] = [groups2[i % len(groups2)] for i in flat_indices]
    else:
        # Handle independent covariates
        for idx, (covariate, groups) in enumerate(covariates.items()):
            covariate_assignments[covariate] = np.random.choice(groups, size=n_cells, p=[p, 1-p])

    # Step 3: Calculate adjusted mean accessibility for each cell and peak
    mean_accessibility = np.tile(base_accessibility, (n_cells, 1))
    effect_sizes = {}
    
    for cdx, (covariate, groups) in enumerate(covariates.items()):
        # Sample features to adjust
        sampled_features_to_adjust = {
            group: np.random.choice([1, 0], size=n_features, p=[sample_feature_p[cdx], 1-sample_feature_p[cdx]])
            for group in groups
        }
        # Generate random adjustment factors for each group
        adjustment = {
            group: minmax_scale(np.random.normal(1,0.5, size = n_features) * np.random.choice((1, -1), size = n_features), feature_range=(1 - max_effect_sizes[cdx], 1 + max_effect_sizes[cdx]))#np.random.uniform(1 - max_effect_sizes[cdx], 1 + max_effect_sizes[cdx], n_features)
            for group in groups
        }
        for i, group in enumerate(covariate_assignments[covariate]):
            # Adjust only the sampled features
            adjustment[group] = np.where(sampled_features_to_adjust[group], adjustment[group], np.ones(n_features))
            # Apply adjustments to the mean accessibility
            mean_accessibility[i] *= adjustment[group]
        
        effect_sizes[covariate] = adjustment

    # Step 4: Simulate fragment counts
    fragment_counts = np.random.poisson(mean_accessibility)

    # Step 5: Compute overlap pivot table for covariates
    covariate_assignments_df = pd.DataFrame(covariate_assignments)
    pivot = covariate_assignments_df.value_counts().reset_index().pivot_table(
        index=covariate_keys[0], columns=covariate_keys[1], #values=0, 
        #aggfunc="sum", 
        fill_value=0
    )
    pivot = pivot / pivot.sum().sum()

    return {
        "data": pd.DataFrame(fragment_counts, columns=[f"peak_{i}" for i in range(n_features)]),
        "effect_sizes": effect_sizes,
        "covariate_assignments": covariate_assignments_df,
        "pivot": pivot
    }


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


def get_DE(adata, effect_sizes):
    sc.pp.log1p(adata)
    
    #Truth
    ef_up = pd.DataFrame(effect_sizes['Uppercase']).sort_values('A')
    ef_low = pd.DataFrame(effect_sizes['Lowercase']).sort_values('a')
    ef_up.index = adata.var.index[ef_up.index]
    ef_low.index = adata.var.index[ef_low.index]
    
    #Cov 1
    sc.tl.rank_genes_groups(adata, "Uppercase", method="wilcoxon")
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    df = pd.DataFrame(
        {
            f"{group}_{key[:1]}": result[key][group]
            for group in groups
            for key in ["names", "logfoldchanges", 'pvals_adj']
        }

    )
    observed_A=df[['A_n', 'A_l', 'A_p']].set_index('A_n')
    observed_B=df[['B_n', 'B_l', 'B_p']].set_index('B_n')
    
    #Cov 2
    sc.tl.rank_genes_groups(adata, "Lowercase", method="wilcoxon")
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    df = pd.DataFrame(
        {
            f"{group}_{key[:1]}": result[key][group]
            for group in groups
            for key in ["names", "logfoldchanges", 'pvals_adj']
        }

    )
    observed_a=df[['a_n', 'a_l', 'a_p']].set_index('a_n')
    observed_b=df[['b_n', 'b_l', 'b_p']].set_index('b_n')
    
    #Prep out
    observed = pd.concat([observed_A, observed_B, observed_a, observed_b], axis = 1)
    observed['order'] = [int(x.split('_')[1]) for x in observed.index]
    observed = observed.sort_values('order')
    
    observed['Eff_up'] = np.log2(effect_sizes['Uppercase']['A'] / effect_sizes['Uppercase']['B'])
    observed['Eff_low'] = np.log2(effect_sizes['Lowercase']['a'] / effect_sizes['Lowercase']['b'])
    
    return observed

def get_model_DE(model, adata):
    DE = compare_groups(model, adata=adata, groupA="A", groupB="B", covariate="Uppercase", add_unknown=True, batch_size=256,exclude_covariates=None,fdr_method="fdr_bh",num_workers=10)
    DE_low = compare_groups(model, adata=adata, groupA="a", groupB="b", covariate="Lowercase", add_unknown=True, batch_size=256,exclude_covariates=None,fdr_method="fdr_bh",num_workers=10)
    return [DE, DE_low]


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix, correction = False)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    if n == 0 or min_dim == 0:
        return 0.0
        
    numerator = chi2 / n
    denominator = min_dim
    if denominator == 0:
        return 0.0
    
    v = np.sqrt(numerator / denominator)
    return v

def find_recall(precision, recall, target_precision):
    valid_indices = np.where(precision >= target_precision)[0]
    if valid_indices.size > 0:
        closest_index = valid_indices[0]
        corresponding_recall = recall[closest_index]
    else:
        corresponding_recall = 'Not found'
        print(f"No recall value found for Precision >= {target_precision}.")
    
    return corresponding_recall
    
from sklearn.preprocessing import minmax_scale

def simulate_scATAC_seq_data_continuous(
    n_cells, n_features, discrete_covariates=None, continuous_covariates=None,
    max_effect_sizes=[0.2, 0.2], sample_feature_p=[0.25, 0.25], continuous_sample_feature_p = None, continuous_max_effect_sizes = None,
    dependency=None, p=0.5, continuous_link_to_groups=None, continuous_entanglement = 0, 
):
    """
    Simulate scATAC-seq data with both discrete and continuous covariates influencing fragment counts.
    
    Parameters:
        n_cells (int): Number of cells.
        n_features (int): Number of genomic peaks/features.
        discrete_covariates (dict, optional): Dictionary where keys are covariate names and values are lists of group labels.
        continuous_covariates (dict, optional): Dictionary where keys are covariate names and values are (min, max) ranges.
        max_effect_sizes (list): Maximum effect sizes for each discrete covariate.
        sample_feature_p (list): Probabilities of selecting features for adjustment for each discrete covariate.
        dependency (np.array or None): Joint probability matrix specifying the dependency between two discrete covariates.
        p (float): Probability of assigning a cell to the first group in each covariate (used if `dependency` is None).
        cont_effect_size (float): Maximum absolute effect size for continuous covariates.

    Returns:
        dict: 
            - 'data': Fragment count matrix (cells x features).
            - 'effect_sizes': Adjustments applied to each covariate group.
            - 'covariate_assignments': Covariate group assignments for each cell.
            - 'pivot': Normalized pivot table showing overlap of discrete covariate assignments.
    """
    # Step 1: Initialize baseline accessibility for each peak
    base_accessibility = np.random.uniform(0.5, 2.0, n_features)
    
    # Step 2: Assign cells to discrete covariate groups
    covariate_assignments = {}
    
    if discrete_covariates:
        covariate_keys = list(discrete_covariates.keys())
        if dependency is not None:
            assert len(covariate_keys) == 2, "Dependency matrix can only be used for two covariates."
            groups1, groups2 = discrete_covariates[covariate_keys[0]], discrete_covariates[covariate_keys[1]]
            flat_indices = np.random.choice(len(dependency.flatten()), size=n_cells, p=dependency.flatten())
            covariate_assignments[covariate_keys[0]] = [groups1[i // len(groups2)] for i in flat_indices]
            covariate_assignments[covariate_keys[1]] = [groups2[i % len(groups2)] for i in flat_indices]
        else:
            for idx, (covariate, groups) in enumerate(discrete_covariates.items()):
                if len(groups) > 2:
                    print(f'cannot use p with more than 2 groups in {covariate}')
                    covariate_assignments[covariate] = np.random.choice(groups, size=n_cells)
                else:
                    covariate_assignments[covariate] = np.random.choice(groups, size=n_cells, p=[p, 1 - p])
    
    # Step 3: Assign continuous covariate values
    if continuous_covariates:
        if continuous_link_to_groups is None:
            for covariate, (low, high) in continuous_covariates.items():
                covariate_assignments[covariate] = np.random.uniform(low, high, size=n_cells)
        else:
            first = 0
            for covariate, (low, high) in continuous_covariates.items():
                if covariate in continuous_link_to_groups.keys() and first == 0:
                    assignment = covariate_assignments[continuous_link_to_groups[covariate]]
                    assignment_unique = list(set(assignment))
                    assignments_cont = minmax_scale(
                        np.random.uniform(low, high, size=len(assignment_unique)),
                        feature_range=(0, 1))
                    mapper = {x:y for x, y in zip(assignment_unique, assignments_cont)}
                    covariate_assignments[covariate] = np.array([mapper[x] for x in assignment])
                    first = 1
                elif covariate in continuous_link_to_groups.keys() and first > 0:
                    to_mod = np.random.choice([1, 0], size=len(assignment_unique), p=[continuous_entanglement, 1 - continuous_entanglement])
                    mod_new = np.random.uniform(low, high, size=len(assignment_unique))
                    assignments_cont_mod = np.where(to_mod, assignments_cont, mod_new)
                    mapper = {x:y for x, y in zip(assignment_unique, assignments_cont_mod)}
                    covariate_assignments[covariate] = np.array([mapper[x] for x in assignment])

                    
                else:
                    covariate_assignments[covariate] = np.random.uniform(low, high, size=n_cells)

    
    # Step 4: Compute adjusted mean accessibility
    mean_accessibility = np.tile(base_accessibility, (n_cells, 1))
    effect_sizes = {}
    
    # Apply discrete covariate effects
    if discrete_covariates:
        for cdx, (covariate, groups) in enumerate(discrete_covariates.items()):
            effect_sizes[covariate] = {}
            sampled_features_to_adjust = {
                group: np.random.choice([1, 0], size=n_features, p=[sample_feature_p[cdx], 1 - sample_feature_p[cdx]])
                for group in groups
            }
            adjustment = {
                group: minmax_scale(
                    np.random.normal(1, 0.5, size=n_features) * np.random.choice((1, -1), size=n_features),
                    feature_range=(1 - max_effect_sizes[cdx], 1 + max_effect_sizes[cdx])
                ) for group in groups
            }
            for i, group in enumerate(covariate_assignments[covariate]):
                adj = np.where(sampled_features_to_adjust[group], adjustment[group], np.ones(n_features))
                mean_accessibility[i] *= adj
                effect_sizes[covariate][group] = adj
    

    # Apply continuous covariate effects (linear scaling)
    if continuous_covariates:
        for cdx, covariate in enumerate(continuous_covariates.keys()):
            effect_sizes[covariate] = {}
            sampled_features_to_adjust = np.random.choice([1, 0], size=n_features, p=[continuous_sample_feature_p[cdx], 1 - continuous_sample_feature_p[cdx]])
            adjustment = minmax_scale(
                    np.random.normal(1, 0.5, size=n_features) * np.random.choice((1, -1), size=n_features),
                    feature_range=(-continuous_max_effect_sizes[cdx], continuous_max_effect_sizes[cdx])
            )
                
            adj = np.where(sampled_features_to_adjust, adjustment, np.zeros(n_features))
            adj_cell = np.tile(adj, (n_cells, 1))
            mean_accessibility *= (2 ** (covariate_assignments[covariate][:,np.newaxis] * adj_cell)) 
            effect_sizes[covariate] = adj

    
    # Step 5: Simulate fragment counts
    fragment_counts = np.random.poisson(mean_accessibility)
    
    # Step 6: Compute overlap pivot table for discrete covariates
    covariate_assignments_df = pd.DataFrame(covariate_assignments)
    pivot = None
    if discrete_covariates:
        pivot = covariate_assignments_df[list(discrete_covariates.keys())].value_counts().reset_index()
    
    return {
        "data": pd.DataFrame(fragment_counts, columns=[f"peak_{i}" for i in range(n_features)]),
        "effect_sizes": effect_sizes,
        "covariate_assignments": covariate_assignments_df,
        "pivot": pivot
    }

def get_model_DE_continuous(model, adata):
    de_2 = compare_groups_de(model, adata, 'cov2')
    de_3 = compare_groups_de(model, adata, 'cov3')
    return [de_2, de_3]

def get_DE_continuous(adata, effect_sizes):
    adata.obs['cov2_group'] = adata.obs.cov2 > 0.5
    adata.obs['cov2_group'] = adata.obs['cov2_group'].map({True:'high2', False:'low2'})
    adata.obs['cov3_group'] = adata.obs.cov3 > 0.5
    adata.obs['cov3_group'] = adata.obs['cov3_group'].map({True:'high3', False:'low3'})

    eff_2 = pd.DataFrame(effect_sizes['cov2'])
    eff_3 = pd.DataFrame(effect_sizes['cov3'])
    eff_2.index = adata.var.index[eff_2.index]
    eff_3.index = adata.var.index[eff_3.index]

    sc.pp.log1p(adata)


    sc.tl.rank_genes_groups(adata, "cov2_group", method="wilcoxon")
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    df = pd.DataFrame(
        {
            f"{group}_{key[:1]}": result[key][group]
            for group in groups
            for key in ["names", "logfoldchanges", 'pvals_adj']
        }

    )
    print(df.head())
    observed_2=df[['high2_n', 'high2_l', 'high2_p']].set_index('high2_n')

    sc.tl.rank_genes_groups(adata, "cov3_group", method="wilcoxon")
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    df = pd.DataFrame(
        {
            f"{group}_{key[:1]}": result[key][group]
            for group in groups
            for key in ["names", "logfoldchanges", 'pvals_adj']
        }

    )
    observed_3=df[['high3_n', 'high3_l', 'high3_p']].set_index('high3_n')

    observed = pd.concat([observed_2, observed_3], axis = 1)
    observed['order'] = [int(x.split('_')[1]) for x in observed.index]
    observed = observed.sort_values('order')

    observed['Eff_cov2'] = eff_2
    observed['Eff_cov3'] = eff_3

    return observed


def simulate_scATAC_seq_data_missing(
    n_cells, n_features, discrete_covariates=None, continuous_covariates=None,
    max_effect_sizes=[0.2, 0.2], sample_feature_p=[0.25, 0.25], continuous_sample_feature_p = None, continuous_max_effect_sizes = None,
    dependency=None, p=0.5, continuous_link_to_groups=None, continuous_entanglement = 0, 
):
    """
    Simulate scATAC-seq data with both discrete and continuous covariates influencing fragment counts.
    
    Parameters:
        n_cells (int): Number of cells.
        n_features (int): Number of genomic peaks/features.
        discrete_covariates (dict, optional): Dictionary where keys are covariate names and values are lists of group labels.
        continuous_covariates (dict, optional): Dictionary where keys are covariate names and values are (min, max) ranges.
        max_effect_sizes (list): Maximum effect sizes for each discrete covariate.
        sample_feature_p (list): Probabilities of selecting features for adjustment for each discrete covariate.
        dependency (np.array or None): Joint probability matrix specifying the dependency between two discrete covariates.
        p (float): Probability of assigning a cell to the first group in each covariate (used if `dependency` is None).
        cont_effect_size (float): Maximum absolute effect size for continuous covariates.

    Returns:
        dict: 
            - 'data': Fragment count matrix (cells x features).
            - 'effect_sizes': Adjustments applied to each covariate group.
            - 'covariate_assignments': Covariate group assignments for each cell.
            - 'pivot': Normalized pivot table showing overlap of discrete covariate assignments.
    """
    # Step 1: Initialize baseline accessibility for each peak
    base_accessibility = np.random.uniform(0.5, 2.0, n_features)
    
    # Step 2: Assign cells to discrete covariate groups
    covariate_assignments = {}
    
    if discrete_covariates:
        covariate_keys = list(discrete_covariates.keys())
        if dependency is not None:
            groups1, groups2 = discrete_covariates[covariate_keys[0]], discrete_covariates[covariate_keys[1]]
            try:
                groups3 = discrete_covariates[covariate_keys[2]]
            except: 
                print('no third')
            flat_indices = np.random.choice(len(dependency.flatten()), size=n_cells, p=dependency.flatten())
            map_mat = np.arange(len(dependency.flatten())).reshape(dependency.shape)
            
            covariate_assignments[covariate_keys[0]] = [groups1[np.where(map_mat == i)[0][0]] for i in flat_indices]
            covariate_assignments[covariate_keys[1]] = [groups2[np.where(map_mat == i)[1][0]] for i in flat_indices]
            try: 
                covariate_assignments[covariate_keys[2]] = [groups3[np.where(map_mat == i)[2][0]] for i in flat_indices]
            except: 
                print('...')
        else:
            for idx, (covariate, groups) in enumerate(discrete_covariates.items()):
                if len(groups) > 2:
                    print(f'cannot use p with more than 2 groups in {covariate}')
                    covariate_assignments[covariate] = np.random.choice(groups, size=n_cells)
                else:
                    covariate_assignments[covariate] = np.random.choice(groups, size=n_cells, p=[p, 1 - p])
    
    # Step 3: Assign continuous covariate values
    if continuous_covariates:
        if continuous_link_to_groups is None:
            for covariate, (low, high) in continuous_covariates.items():
                covariate_assignments[covariate] = np.random.uniform(low, high, size=n_cells)
        else:
            first = 0
            for covariate, (low, high) in continuous_covariates.items():
                if covariate in continuous_link_to_groups.keys() and first == 0:
                    assignment = covariate_assignments[continuous_link_to_groups[covariate]]
                    assignment_unique = list(set(assignment))
                    assignments_cont = minmax_scale(
                        np.random.uniform(low, high, size=len(assignment_unique)),
                        feature_range=(0, 1))
                    mapper = {x:y for x, y in zip(assignment_unique, assignments_cont)}
                    covariate_assignments[covariate] = np.array([mapper[x] for x in assignment])
                    first = 1
                elif covariate in continuous_link_to_groups.keys() and first > 0:
                    to_mod = np.random.choice([1, 0], size=len(assignment_unique), p=[continuous_entanglement, 1 - continuous_entanglement])
                    mod_new = np.random.uniform(low, high, size=len(assignment_unique))
                    assignments_cont_mod = np.where(to_mod, assignments_cont, mod_new)
                    mapper = {x:y for x, y in zip(assignment_unique, assignments_cont_mod)}
                    covariate_assignments[covariate] = np.array([mapper[x] for x in assignment])

                    
                else:
                    covariate_assignments[covariate] = np.random.uniform(low, high, size=n_cells)

    
    # Step 4: Compute adjusted mean accessibility
    mean_accessibility = np.tile(base_accessibility, (n_cells, 1))
    effect_sizes = {}
    
    # Apply discrete covariate effects
    if discrete_covariates:
        for cdx, (covariate, groups) in enumerate(discrete_covariates.items()):
            effect_sizes[covariate] = {}
            sampled_features_to_adjust = {
                group: np.random.choice([1, 0], size=n_features, p=[sample_feature_p[cdx], 1 - sample_feature_p[cdx]])
                for group in groups
            }
            adjustment = {
                group: minmax_scale(
                    np.random.normal(1, 0.5, size=n_features) * np.random.choice((1, -1), size=n_features),
                    feature_range=(1 - max_effect_sizes[cdx], 1 + max_effect_sizes[cdx])
                ) for group in groups
            }
            for i, group in enumerate(covariate_assignments[covariate]):
                adj = np.where(sampled_features_to_adjust[group], adjustment[group], np.ones(n_features))
                mean_accessibility[i] *= adj
                effect_sizes[covariate][group] = adj
    

    # Apply continuous covariate effects (linear scaling)
    if continuous_covariates:
        for cdx, covariate in enumerate(continuous_covariates.keys()):
            effect_sizes[covariate] = {}
            sampled_features_to_adjust = np.random.choice([1, 0], size=n_features, p=[continuous_sample_feature_p[cdx], 1 - continuous_sample_feature_p[cdx]])
            adjustment = minmax_scale(
                    np.random.normal(1, 0.5, size=n_features) * np.random.choice((1, -1), size=n_features),
                    feature_range=(-continuous_max_effect_sizes[cdx], continuous_max_effect_sizes[cdx])
            )
                
            adj = np.where(sampled_features_to_adjust, adjustment, np.zeros(n_features))
            adj_cell = np.tile(adj, (n_cells, 1))
            #Scale on lfc
            mean_accessibility *= (2 ** (covariate_assignments[covariate][:,np.newaxis] * adj_cell)) 
            effect_sizes[covariate] = adj

    
    # Step 5: Simulate fragment counts
    fragment_counts = np.random.poisson(mean_accessibility)
    
    # Step 6: Compute overlap pivot table for discrete covariates
    covariate_assignments_df = pd.DataFrame(covariate_assignments)
    pivot = None
    if discrete_covariates:
        pivot = covariate_assignments_df[list(discrete_covariates.keys())].value_counts().reset_index()
    
    return {
        "data": pd.DataFrame(fragment_counts, columns=[f"peak_{i}" for i in range(n_features)]),
        "effect_sizes": effect_sizes,
        "covariate_assignments": covariate_assignments_df,
        "pivot": pivot
    }


def compare_groups(self, adata, covariate, groupA, groupB = None, exclude_covariates = None, batch_size = 256, num_workers = 10, fdr_method = 'fdr_bh', add_unknown = True):
    ## Checks
    # Covariate
    all_covariates = self.discrete_covariate_names
    if covariate not in all_covariates:
        raise ValueError(f"Covariate must be one of {all_covariates}.")

    ## GroupA
    covariate_levels = list(self.data_register['covars_dict'][covariate].keys())[:-1]
    if groupA not in covariate_levels:
        raise ValueError(f"GroupA must be one of {covariate_levels}.")

    ## Target
    if groupB is not None:
        groupB_covariate_levels = list(set(covariate_levels) - set([groupA]))
        if groupB not in groupB_covariate_levels:
            raise ValueError(f"GroupB must be one of {groupB_covariate_levels} or None.")

    ## Get adata object for each group
    adata_groupA = adata[adata.obs[covariate].isin([groupA]),:]
    if groupB is not None:
        adata_groupB = adata[adata.obs[covariate].isin([groupB]),:]
    else:
        adata_groupB = adata[~adata.obs[covariate].isin([groupA]),:]

    ## Setup covariates to add
    covars_to_add = self.discrete_covariate_names + self.continuous_covariate_names
    if exclude_covariates is not None:
        covars_to_add = list(set(covars_to_add) - set(exclude_covariates))

    ## Reconstruction
    # GroupA
    groupA_recon = torch.stack([model.known.decoder_list[x](model.known.discrete_covariates_embeddings[covariate](torch.tensor(model.known.data_register['covars_dict'][covariate][groupA], device = 'cuda'))) for x in range(self.n_decoders)]).detach().cpu().numpy()


    # GroupB
    groupB_recon = torch.stack([model.known.decoder_list[x](model.known.discrete_covariates_embeddings[covariate](torch.tensor(model.known.data_register['covars_dict'][covariate][groupB], device = 'cuda'))) for x in range(self.n_decoders)]).detach().cpu().numpy()


    ## Process
    # Stats per decoder
    diff, Z, P = [], [], []
    for decoder in range(self.n_decoders):
        diff_tmp = groupA_recon[decoder,:] - groupB_recon[decoder,:]
        Z_tmp = (diff_tmp - diff_tmp.mean()) / diff_tmp.std()
        P_tmp = 2 * (1 - norm.cdf(np.abs(Z_tmp)))
        diff.append(diff_tmp)
        Z.append(Z_tmp)
        P.append(P_tmp)

    # Stacks
    P = np.stack(P, axis = 0)
    Z = np.stack(Z, axis = 0)
    diff = np.stack(diff, axis = 0)

    # Summary
    chi_squared_stat  = -2 * np.sum(np.log(P), axis = 0)
    df = 2 * self.n_decoders
    P_combined = 1 - chi2.cdf(chi_squared_stat, df)
    Z_combined = np.mean(Z, axis = 0)
    diff_combined = np.mean(diff, axis = 0)
    reject, pvals_corrected, _, _ = multipletests(P_combined, alpha=0.05, method=fdr_method)

    # Setup results
    test_res = pd.DataFrame(
        {
            'Feature': adata.var_names, 
            'groupA': groupA_recon.mean(axis = 0),
            'groupB': groupB_recon.mean(axis = 0),
            'Difference': diff_combined,
            'Zscore': Z_combined, 
            'Pvalue': P_combined,
            'FDR': pvals_corrected
        }
    )

    return test_res