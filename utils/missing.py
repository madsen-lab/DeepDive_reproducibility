def predict_missing(self, adata, covar_importance = None, group_by = None, vote_for = [], majority_vote = False, metric = 'nll', mode = 'ablate', covariates = None, n_steps = 20, predict_mode = 'selected', add_unknown = False, covars_to_add = None, verbose = True, continous_window = 1.2, select = None, batch_size = 256): 
    # Take a copy
    adata = adata.copy()

    # Covariate importance
    if covar_importance is None:
        if verbose:
            print("Calculating covariate importance.")
        covar_importance = self.covariate_importance()

    # Covariates to predict
    if covariates is None:
        covariates = self.discrete_covariate_names + self.continuous_covariate_names

    # Subset covariance importance by selected covariants
    covar_importance = covar_importance[covar_importance['Covariate'].isin(covariates)]

    # Loop across covariates
    for covar in covar_importance['Covariate']:
        covar_index = np.argwhere(adata.obs.columns == covar)[0][0]
        if covar in self.discrete_covariate_names:
            if covar in list(self.unknown_keys.keys()):
                mask_key = self.unknown_keys[covar]
            else:
                mask_key = 'Training_mask'
            levels = list(set(self.data_register['covariate_names_unique'][covar]) - set([mask_key]))
        else:
            mask_key = self.continous_mask_value
            levels = list(np.linspace(self.data_register['continuous_covariate_scalers'][covar].min.numpy() / continous_window, self.data_register['continuous_covariate_scalers'][covar].max.numpy() * continous_window, num = n_steps)[:,0])

        if np.sum(adata.obs[covar] == mask_key) > 0:
            indices = list(np.argwhere((adata.obs[covar] == mask_key).values)[:,0])
            if verbose:
                print("Predicting " + str(covar) + " for " + str(len(indices)) + " cells missing that covariate.")
            adata_cell = adata[indices,:].copy()
            counter = 0
            for level in levels:
                adata_cell.obs.iloc[:,covar_index] = level
                rec = self.predict(adata_cell, predict_mode=predict_mode, add_unknown = add_unknown, covars_to_add = covars_to_add, batch_size = batch_size)
                if select is not None:
                    err_level = self._calc_metric(rec[:,select], adata_cell[:,select], metric, axis = 1)[:,np.newaxis] 
                else:
                    err_level = self._calc_metric(rec, adata_cell, metric, axis = 1)[:,np.newaxis]                             
                if counter == 0:
                    err = err_level
                else:
                    err = np.concatenate((err, err_level),axis=1)
                counter += 1

            inferred = np.argmin(err,axis=1)
            mapped = np.array([levels[i] for i in inferred])
            adata.obs['Error'] = 0
            adata.obs['ErrorMin'] = 0
            adata.obs['ErrorMax'] = 0
            for indice in range(len(indices)):
                adata.obs.iloc[indices[indice],covar_index] = mapped[indice]
                adata.obs.iloc[indices[indice],np.argwhere(adata.obs.columns == 'Error')[0][0]] = (np.max(err,axis=1)[indice] - np.min(err,axis=1)[indice]) / np.max(err,axis=1)[indice]
                adata.obs.iloc[indices[indice],np.argwhere(adata.obs.columns == 'ErrorMax')[0][0]] = np.max(err,axis=1)[indice]
                adata.obs.iloc[indices[indice],np.argwhere(adata.obs.columns == 'ErrorMin')[0][0]] = np.min(err,axis=1)[indice]

            if majority_vote:
                if covar in vote_for:
                    if verbose:
                        print("Majority voting for " + str(covar) + " grouping by " + str(group_by) + ".")
                    adata = self.majority_voting(adata, group_by, [covar])
    return adata

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def train_xgb_from_adata(
    adata_train, adata_test, label_key="cell_label", 
    n_estimators=200, max_depth=6, learning_rate=0.1, 
    subsample=0.8, colsample_bytree=0.8, random_state=0
):
    """
    Train an XGBoost model on TF-IDF transformed AnnData to predict cell labels.

    Parameters
    ----------
    adata_train : AnnData
        Training AnnData object (with TF-IDF in .X).
    adata_test : AnnData
        Testing AnnData object (with TF-IDF in .X).
    label_key : str
        Column in adata.obs with labels.
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree
        Standard XGBoost hyperparameters.
    random_state : int
        Reproducibility.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    y_true, y_pred : np.ndarray
        True and predicted labels for the test set.
    encoder : LabelEncoder
        Encoder mapping classes to integers.
    """

    # Extract features
    X_train = adata_train.X
    X_test = adata_test.X

    # Extract labels
    y_train = adata_train.obs[label_key].values
    y_test = adata_test.obs[label_key].values

    # Encode labels as integers for XGBoost
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    n_classes = len(encoder.classes_)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes for classification, found {n_classes}")

    # Define model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softmax" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="hist"
    )

    # Fit
    model.fit(X_train, y_train_enc)

    # Predict
    y_pred_enc = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_enc)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return model, y_test, y_pred, encoder, accuracy_score(y_test, y_pred) 
