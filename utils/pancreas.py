import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, average_precision_score, accuracy_score
import math
from tqdm.notebook import tqdm, trange
from typing import Union
from anndata import AnnData
from mudata import MuData
from scipy import sparse
from multiprocessing import Pool, cpu_count
import logging
import torch
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import scipy
import xgboost as xgb


def covariate_importance(model, cell_type_covar):
    # Set model into evaluation mode
    model.eval()
    res_dict = {}

    # Get all covariates
    all_covars = model.discrete_covariate_names + model.continuous_covariate_names
    all_covars.remove(cell_type_covar)
    all_covars.remove('donor')
    all_covars.remove('Center')
    all_covars.remove('Storage')
    all_covars.remove('Purity')
    all_covars.remove('map_frac')
    
    for cdx, c in enumerate(model.data_register['covariate_names_unique'][cell_type_covar][:-1]):
        base = model.known.discrete_covariates_embeddings[cell_type_covar].weight[cdx]
        
        # Loop across covariates in the model
        counter = 0
        for covar in all_covars:
            # Loop across decoders
            importance = []
            for decoder in range(model.n_decoders):
                if covar in model.discrete_covariate_names:
                    discrete = True
                    logits = model.known.decoder_list[decoder](base + model.known.discrete_covariates_embeddings[covar].weight).detach().cpu().numpy()
                    nlevels = len(model.data_register['covariate_names_unique'][covar])-1
                    logits = logits[:nlevels,:]
                else:
                    discrete = False
                    weights = model.known.continuous_covariates_embeddings[covar].weight
                    high_logits = model.known.decoder_list[decoder](base + weights).detach().cpu().numpy()
                    low_logits = model.known.decoder_list[decoder](base + torch.zeros_like(weights)).detach().cpu().numpy()
                    logits = np.concatenate((high_logits,low_logits))
                    
                importance.append(logits[:2,:].max(axis=0) - logits[:2,:].min(axis=0))

            # Average across multiple decoder or extract a single decoder
            if len(importance) > 1:
                importance = np.mean(np.abs(np.vstack(importance).mean(axis=0)))
            else:
                importance = np.mean(np.abs(importance[0]))

            # Setup results
            res_tmp = pd.DataFrame({'Covariate': covar, 'Discrete': discrete, 'Importance': importance}, index = [counter])

            # Combine results
            if counter == 0:
                res = res_tmp
            else:
                res = pd.concat((res, res_tmp))
            counter += 1

        res.sort_values(by='Importance', ascending=False, inplace=True)
        res_dict[c] = res
    return res_dict

# Define function to calculate metric
def calculate_metric(model, truth, prediction, metric):
    if metric == "Kappa":
        return cohen_kappa_score(truth, (prediction > 0.5).astype(int))
    elif metric == "Accuracy":
        return accuracy_score(truth, (prediction > 0.5).astype(int))
    elif metric == "AP":
        return average_precision_score(truth, prediction)

    # Predicts
    y_train_pred = model.predict(X_train)
    y_eval_pred = model.predict(X_eval)
    y_test_pred = model.predict(X_test)

    # Make dictionary
    predictions = {
        'Train': {'Truth': y_train, 'Prediction': y_train_pred},
        'Eval': {'Truth': y_eval, 'Prediction': y_eval_pred},
        'Test': {'Truth': y_test, 'Prediction': y_test_pred}
    }

    # Calculate metrics
    ds, met, score = [], [], []
    for dataset in predictions.keys():
        for metric in ['Kappa', 'Accuracy','AP']:
            score.append(calculate_metric(predictions[dataset]['Truth'], predictions[dataset]['Prediction'], metric))
            ds.append(dataset)
            met.append(metric)

    # Create pandas object
    results = pd.DataFrame({
        'Dataset': ds,
        'Metric': met,
        'Score': score
    })

def counterfactual_prediction(
    model,
    adata_object,
    covar_group_map,
    classifier,
    features,
    reference,
    covars_to_add=['celltype', 'Disease', 'Gender', 'Race', 'Age', 'BMI', 'HbA1c'],
    add_unknown=True,
    library_size="observed",
    batch_size=256,
    num_workers=1,
    use_decoder="all",
    get_counts = False,
):

    model.eval()

    adata = adata_object.copy()

    all_covariates = model.discrete_covariate_names + model.continuous_covariate_names
    for covariate in covar_group_map.keys():
        if covariate not in all_covariates:
            raise ValueError(
                f"The covariate {covariate} is not part of covariates used during training: {all_covariates}."
            )

    for covariate, group in covar_group_map.items():
        adata.obs[covariate + "_org"] = adata.obs[covariate].copy()
        if covariate in model.discrete_covariate_names:
            adata.obs[covariate] = group
        else:
            adata.obs[covariate] += group
    
    
    recon = model.predict(
        adata,
        covars_to_add=covars_to_add,
        batch_size=batch_size,
        num_workers=num_workers,
        add_unknown=add_unknown,
        use_decoder=use_decoder,
        library_size=10_000,
        predict_mode = "selected"
    )
    
    if get_counts:
        return recon
    else:
        # Subset and predict
        X = xgb.QuantileDMatrix(np.array(recon.X)[:,features], max_bin = 15, ref=reference)
        pred = classifier.predict(X)
        return pd.Series(pred, index = adata.obs.index)

    
def compare_groups_de(
    model,
    adata,
    covariate,
    groupA=None,
    groupB=None,
    fdr_method="fdr_bh",
    background = None
):
    model.eval()

    ## Checks
    # Covariate
    all_covariates = model.discrete_covariate_names + model.continuous_covariate_names
    if covariate not in all_covariates:
        raise ValueError(f"Covariate must be one of {all_covariates}.")
    else:
        if covariate in model.discrete_covariate_names:
            covar_type = "discrete"
        else:
            covar_type = "continuous"

    if (covar_type == "discrete" and groupA is None) or (
        covar_type == "discrete" and groupB is None
    ):
        raise ValueError(
            f"If using a discrete covariate, please set both groupA and groupB."
        )
    if groupA is not None:
        if groupA == groupB:
            raise ValueError(f"roupA and groupB must be different.")

    ## GroupA
    if groupA is not None:
        covariate_levels = list(model.data_register["covars_dict"][covariate].keys())[
            :-1
        ]
        if groupA not in covariate_levels:
            raise ValueError(f"GroupA must be one of {covariate_levels}.")

    ## GroupB
    if groupB is not None:
        groupB_covariate_levels = list(set(covariate_levels) - set([groupA]))
        if groupB not in groupB_covariate_levels:
            raise ValueError(f"GroupB must be one of {groupB_covariate_levels}.")
            

    if covar_type == "discrete":
        embedding = model.known.discrete_covariates_embeddings[covariate]
        embedding_idx_A = torch.tensor(
            model.known.data_register["covars_dict"][covariate][groupA], device=model.device
        )
        embedding_idx_B = torch.tensor(
            model.known.data_register["covars_dict"][covariate][groupB], device=model.device
        )
        embedding_A = embedding(embedding_idx_A)
        embedding_B = embedding(embedding_idx_B)
    else:
        embedding = model.known.continuous_covariates_embeddings[covariate]
        embedding_idx = torch.tensor(0, device=model.device)
        embedding_A = embedding(embedding_idx) 
        embedding_B = torch.zeros_like(embedding_A)
        
        
    base = torch.zeros_like(embedding_A)
    if background is not None:
        
        for key, value in background.items():
            if key in model.discrete_covariate_names:
                embedding = model.known.discrete_covariates_embeddings[key]
                embedding_idx = torch.tensor(
                    model.known.data_register["covars_dict"][key][value], device=model.device
                )
                
                base += embedding(embedding_idx)
                
            else:
                embedding = model.known.continuous_covariates_embeddings[key]
                embedding_idx = torch.tensor(0, device=model.device)
                base += embedding(embedding_idx) * value
                
            
    # GroupA
    groupA_recon = (
        torch.stack(
            [model.known.decoder_list[x](base + embedding_A) for x in range(model.n_decoders)]
        )
        .detach()
        .cpu()
        .numpy() 
    ) 
    # GroupB
    groupB_recon = (
        torch.stack(
            [model.known.decoder_list[x](base + embedding_B) for x in range(model.n_decoders)]
        )
        .detach()
        .cpu()
        .numpy()
    ) 

    ## Process
    # Stats per decoder
    diff, Z, P = [], [], []
    for decoder in range(model.n_decoders):
        diff_tmp = groupA_recon[decoder, :] - groupB_recon[decoder, :]
        Z_tmp = (diff_tmp - diff_tmp.mean()) / diff_tmp.std()
        P_tmp = 2 * (1 - norm.cdf(np.abs(Z_tmp)))
        diff.append(diff_tmp)
        Z.append(Z_tmp)
        P.append(P_tmp)

    # Stacks
    P = np.stack(P, axis=0)
    Z = np.stack(Z, axis=0)
    diff = np.stack(diff, axis=0)

    # Summary
    chi_squared_stat = -2 * np.sum(np.log(P), axis=0)
    df = 2 * model.n_decoders
    P_combined = 1 - chi2.cdf(chi_squared_stat, df)
    Z_combined = np.mean(Z, axis=0)
    diff_combined = np.mean(diff, axis=0)
    reject, pvals_corrected, _, _ = multipletests(
        P_combined, alpha=0.05, method=fdr_method
    )

    # Setup results
    test_res = pd.DataFrame(
        {
            "Feature": adata.var_names,
            "groupA": groupA_recon.mean(axis=0),
            "groupB": groupB_recon.mean(axis=0),
            "Difference": diff_combined,
            "Zscore": Z_combined,
            "Pvalue": P_combined,
            "FDR": pvals_corrected,
        }
    )

    return test_res


############################################################    
###Modified from https://github.com/pinellolab/pychromVAR###
############################################################
def find_motifs(adata, motifs, alphabet=['A', 'C', 'G', 'T'], reverse_complement=True, bin_size=0.1, eps=0.0001, threshold=0.0001):
    # Process motifs
    log_threshold = math.log2(threshold)
    motifs_ = read_meme(motifs)
    motifs_fwd: list[tuple[str, np.ndarray]] = []
    motifs_rev: list[tuple[str, np.ndarray]] = []
    for name in motifs_:
        pwm = motifs_[name]
        motifs_fwd.append((name, pwm))
        if reverse_complement:
            motifs_rev.append((name + "-rc", pwm[::-1, ::-1]))

    if reverse_complement:
        motifs = [*motifs_fwd, *motifs_rev]
    else:
        motifs = motifs_fwd

    n_motifs = len(motifs)

    motif_names = np.array([name for name, _ in motifs])
    motif_lengths = [0] + [pwm.shape[-1] for _, pwm in motifs]
    motif_lengths = np.cumsum(motif_lengths).astype(np.uint64)

    motif_pwms = np.concatenate([pwm for _, pwm in motifs], axis=-1)
    motif_pwms = np.log2(motif_pwms + eps) - math.log2(0.25)

    _smallest, _score_to_pvals = _all_pwm_to_mapping(motif_pwms, motif_lengths, bin_size)
    _score_to_pvals_lengths = [0]
    _score_thresholds = np.empty(n_motifs, dtype=np.float32)

    for i in range(n_motifs):
        _score_to_pvals_lengths.append(len(_score_to_pvals[i]))
        idx = np.where(_score_to_pvals[i] < log_threshold)[0]
        if len(idx) > 0:
            _score_thresholds[i] = (idx[0] + _smallest[i]) * bin_size                              
        else:
            _score_thresholds[i] = float("inf")

    _score_to_pvals = np.concatenate(_score_to_pvals)
    _score_to_pvals_lengths = np.cumsum(_score_to_pvals_lengths)

    # Process sequences
    X, lengths = [], [0]
    alphabet = ''.join(alphabet)
    alpha_idxs = np.frombuffer(bytearray(alphabet, 'utf8'), dtype=np.int8)
    one_hot_mapping = np.zeros(256, dtype=np.int8) - 1
    for i, idx in enumerate(alpha_idxs):
        one_hot_mapping[idx] = i

    sequences = adata.uns['peak_seq']
    for i in range(len(sequences)):
        chrom = sequences[i].upper()
        lengths.append(lengths[-1] + len(chrom))
        X_idxs = np.frombuffer(bytearray(chrom, "utf8"), dtype=np.int8)
        _fast_convert(X_idxs, one_hot_mapping)
        X.append(X_idxs)

    X = np.concatenate(X)
    X_lengths = np.array(lengths, dtype=np.int64)

    # Get hits
    hits = _fast_hits(X, X_lengths, motif_pwms, motif_lengths, _score_thresholds, bin_size, _smallest, _score_to_pvals, _score_to_pvals_lengths)

    # Convert the results to pandas DataFrames
    names = ['sequence_name', 'start', 'end', 'score', 'p-value']
    n_ = n_motifs // 2 if reverse_complement else n_motifs

    for i in range(n_):
        if reverse_complement:
            hits_ = pd.DataFrame(hits[i] + hits[i + n_], columns=names)
            hits_['strand'] = ['+'] * len(hits[i]) + ['-'] * len(hits[i+n_])
        else:
            hits_ = pd.DataFrame(hits[i], columns=names)
            hits_['strand'] = ['+'] * len(hits[i])

        hits_['motif_name'] = [motif_names[i] for _ in range(len(hits_))]
        hits_['motif_idx'] = np.ones(len(hits_), dtype='int64') * i

        hits[i] = hits_[['motif_name', 'motif_idx', 'sequence_name', 'start', 'end', 'strand', 'score', 'p-value']]

    hits = hits[:n_]

    counter = 0
    for i in range(len(hits)):
        if len(hits[i]) > 0:
            counter += 1

    hit_arr = np.zeros((adata.shape[1],counter), dtype=np.uint8)
    names = []
    counter = 0
    for i in range(len(hits)):
        if len(hits[i]) > 0:
            hit_arr[hits[i]['sequence_name'],counter] = 1
            names.append(hits[i]['motif_name'][0].split(" ")[0])
            counter += 1
            
    # Finalize adata
    adata = adata.copy()
    adata.varm['motif_match'] = hit_arr
    adata.uns['motif_name'] = names
    return adata

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def compute_deviations(data: Union[AnnData, MuData], n_jobs=-1, chunk_size:int=10000, gpu = False) -> AnnData:
    """Compute raw and bias-corrected deviations.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    n_jobs : int, optional
        Number of cpus used for motif matching. If set to -1, all cpus will be used. Default: -1.

    Returns
    -------
    Anndata
        An anndata object containing estimated deviations.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")
    # check if the object contains bias in Anndata.varm
    assert "bg_peaks" in adata.varm_keys(
    ), "Cannot find background peaks in the input object, please first run get_bg_peaks!"
    expectation_obs, expectation_var = compute_expectation(count=adata.X, return_torch = gpu)

    # compute background deviations for bias-correction
    n_bg_peaks = adata.varm['bg_peaks'].shape[1]
    if gpu:
        motif_match = torch.tensor(adata.varm['motif_match'], device = 'cuda', dtype=torch.float32)
        obs_dev = torch.zeros(size=(adata.n_obs, motif_match.shape[1]), dtype=torch.float32, device = 'cuda')
        bg_dev = torch.zeros(size=(n_bg_peaks, adata.n_obs, len(
            adata.uns['motif_name'])), dtype=torch.float32, device = 'cuda')
    else:
        motif_match = adata.varm['motif_match']
        obs_dev = np.zeros((adata.n_obs, motif_match.shape[1]), dtype=np.float32)
        bg_dev = np.zeros(shape=(n_bg_peaks, adata.n_obs, len(
            adata.uns['motif_name'])), dtype=np.float32)

    ### instead of iterating over bg peaks, iterate over X
    for item in adata.chunked_X(chunk_size):
        X, start, end = item
        if gpu:
            if sparse.issparse(X):
                X = X.todense()
            X = torch.tensor(X, device = 'cuda', dtype=torch.float32)
            obs_dev[start:end, :] = _compute_deviations_gpu((motif_match, X, expectation_obs[start:end], expectation_var))
        else: 
            obs_dev[start:end, :] = _compute_deviations((motif_match, X, expectation_obs[start:end], expectation_var))

        for i in range(n_bg_peaks):
            if gpu:
                bg_peak_idx = adata.varm['bg_peaks'][:, i]
                bg_motif_match = torch.tensor(adata.varm['motif_match'][bg_peak_idx, :], device = 'cuda', dtype=torch.float32)
                bg_dev[i, start:end, :] = _compute_deviations_gpu((bg_motif_match, X, expectation_obs[start:end], expectation_var))
            else:
                bg_peak_idx = adata.varm['bg_peaks'][:, i]
                bg_motif_match = adata.varm['motif_match'][bg_peak_idx, :]
                bg_dev[i, start:end, :] = _compute_deviations((bg_motif_match, X, expectation_obs[start:end], expectation_var))
    if gpu:
        mean_bg_dev = torch.mean(bg_dev, axis = 0)
        std_bg_dev = torch.std(bg_dev, axis=0)
        dev = (obs_dev - mean_bg_dev) / std_bg_dev
        mean_bg_dev = mean_bg_dev.cpu().numpy()
        std_bg_dev = std_bg_dev.cpu().numpy()
        dev = dev.cpu().numpy()

    else:
        mean_bg_dev = np.mean(bg_dev, axis=0)
        std_bg_dev = np.std(bg_dev, axis=0)
        dev = (obs_dev - mean_bg_dev) / std_bg_dev
    dev = np.nan_to_num(dev, 0)
    dev = AnnData(dev, dtype=np.float32)
    dev.obs_names = adata.obs_names
    dev.var_names = adata.uns['motif_name']
    return dev.copy()


def _compute_deviations(arguments):
    motif_match, count, expectation_obs, expectation_var = arguments
    ### motif_match: n_var x n_motif
    ### count, exp: n_obs x n_var
    observed = count.dot(motif_match)
    expected = expectation_obs.dot(expectation_var.dot(motif_match))
    if sparse.issparse(observed):
        observed = observed.todense()
    if sparse.issparse(expected):
        expected = expected.todense()
    out = np.zeros(expected.shape, dtype=expected.dtype)
    np.divide(observed - expected, expected, out=out, where=expected != 0)
    return out

def _compute_deviations_gpu(arguments):
    motif_match, count, expectation_obs, expectation_var = arguments
    ### motif_match: n_var x n_motif
    ### count, exp: n_obs x n_var
    observed = torch.matmul(count, motif_match)
    
    expected = torch.matmul(expectation_obs, torch.matmul(expectation_var, motif_match))
    out = torch.zeros(expected.shape, dtype=expected.dtype, device = 'cuda')
    torch.div(observed - expected, expected, out=out)
    out[expected == 0] = 0
    return out

def compute_expectation(count: Union[np.array, sparse.csr_matrix], return_torch = False) -> np.array:
    """
    Compute expetation accessibility per peak and per cell by assuming
    identical read probability per peak for each cell with a sequencing
    depth matched to that cell observed sequencing depth

    Parameters
    ----------
    count : Union[np.array, sparse.csr_matrix]
        Count matrix containing raw accessibility data.

    Returns
    -------
    np.array, np.array
        Expectation matrix pair when multiplied gives
    """
    a = np.asarray(count.sum(0), dtype=np.float32).reshape((1, count.shape[1]))
    a /= a.sum()
    b = np.asarray(count.sum(1), dtype=np.float32).reshape((count.shape[0], 1))
    if return_torch:
        return torch.tensor(b, device = 'cuda'), torch.tensor(a, device = 'cuda')
    else:
        return b, a

def permutation_test(
    adata: AnnData,
    groupby: str,
    group1: str,
    group2=None,
    n_iterations: int = 1000,
):
    adata = adata.copy()
    obs_stat = np.array(adata.X[adata.obs[groupby] == group1].mean(axis=0) - adata.X[adata.obs[groupby] == group2].mean(axis=0)).ravel()
    bg_stat = np.zeros((n_iterations, obs_stat.shape[0]))
    group1_mean = adata.X[adata.obs[groupby] == group1].mean(axis=0).ravel()
    group2_mean = adata.X[adata.obs[groupby] == group2].mean(axis=0).ravel()
    
    for i in range(n_iterations):
        adata.obs[groupby] = np.random.permutation(adata.obs[groupby].values)
        bg_stat[i, :] = np.array(adata.X[adata.obs[groupby] == group1].mean(axis=0) - adata.X[adata.obs[groupby] == group2].mean(axis=0)).ravel()
    bg_stat_mean = np.mean(bg_stat, axis=0)
    bg_stat_std = np.std(bg_stat, axis=0)

    zscore = (obs_stat - bg_stat_mean) / bg_stat_std

    pval = scipy.stats.norm.cdf(zscore, 0, 1)
    pval = np.minimum(pval, 1 - pval)

    res = pd.DataFrame(
        {
            f"{group1}_mean": group1_mean,
            f"{group2}_mean": group2_mean,
            "stat": obs_stat,
            "zscore": zscore,
            "pval": pval,
        },
        index=adata.var_names,
    )

    return res

