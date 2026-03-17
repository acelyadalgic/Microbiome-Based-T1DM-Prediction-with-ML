import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import pyswarms as ps
import warnings

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_DIR = "input_path"
OUTPUT_METRICS = "pooled_within_country_scaled_metrics.csv"
OUTPUT_FEATURES = "pooled_within_country_scaled_features.csv"

PSO_PARTICLES = 30
PSO_ITERS = 50
def pso_feature_selection(X_train, y_train):
    np.random.seed(RANDOM_STATE)
    n_features = X_train.shape[1]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def fitness_function(swarm_position):
        fitness_scores = []
        for particle in swarm_position:
            binary_mask = np.round(particle)
            selected_idx = np.where(binary_mask == 1)[0]
            if len(selected_idx) == 0:
                fitness_scores.append(1.0); continue

            aucs = []
            for train_idx, val_idx in inner_cv.split(X_train, y_train):
                clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE, n_jobs=1)
                clf.fit(X_train.iloc[train_idx, selected_idx], y_train.iloc[train_idx])
                probs = clf.predict_proba(X_train.iloc[val_idx, selected_idx])[:, 1]
                aucs.append(roc_auc_score(y_train.iloc[val_idx], probs))
            
            mean_auc = np.mean(aucs)
            alpha = 0.5 
            fitness = (1 - mean_auc) + alpha * (len(selected_idx) / n_features)
            fitness_scores.append(fitness)
        return np.array(fitness_scores)

    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 2, 'p': 2}
    optimizer = ps.discrete.BinaryPSO(n_particles=PSO_PARTICLES, dimensions=n_features, options=options)
    cost, pos = optimizer.optimize(fitness_function, iters=PSO_ITERS, verbose=False)
    return np.where(np.round(pos) == 1)[0]
def run_fold(model_name, repeat, fold, train_idx, test_idx, X_full, y_full, countries_full):
    X_tr_raw, X_te_raw = X_full.iloc[train_idx].copy(), X_full.iloc[test_idx].copy()
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
    c_train, c_test = countries_full.iloc[train_idx], countries_full.iloc[test_idx]
    keep_prev = (X_tr_raw > 0).mean(axis=0) >= 0.10
    X_tr_raw = X_tr_raw.loc[:, keep_prev]
    X_te_raw = X_te_raw.loc[:, keep_prev]

    keep_var = X_tr_raw.var(axis=0) > 0.001
    X_tr_raw = X_tr_raw.loc[:, keep_var]
    X_te_raw = X_te_raw.loc[:, keep_var]
    X_train_scaled = pd.DataFrame(index=X_tr_raw.index, columns=X_tr_raw.columns)
    X_test_scaled = pd.DataFrame(index=X_te_raw.index, columns=X_te_raw.columns)

    for country in c_train.unique():
        m_tr = (c_train == country)
        m_te = (c_test == country)
        
        scaler = StandardScaler()
        X_train_scaled.loc[m_tr] = scaler.fit_transform(X_tr_raw.loc[m_tr])
        if m_te.any():
            X_test_scaled.loc[m_te] = scaler.transform(X_te_raw.loc[m_te])

    X_train_scaled = X_train_scaled.apply(pd.to_numeric)
    X_test_scaled = X_test_scaled.apply(pd.to_numeric)
    selected_idx = pso_feature_selection(X_train_scaled, y_train)
    selected_features = X_train_scaled.columns[selected_idx].tolist()
    if not selected_features: return None
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    search = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), 
                                param_distributions=param_dist, n_iter=10, cv=3, scoring='roc_auc', n_jobs=1)
    search.fit(X_train_scaled[selected_features], y_train)
    best_clf = search.best_estimator_
    y_prob = best_clf.predict_proba(X_test_scaled[selected_features])[:, 1]
    y_pred = best_clf.predict(X_test_scaled[selected_features])

    metrics = {
        "Model": model_name, "Repeat": repeat, "Fold": fold,
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Features_Count": len(selected_features)
    }
    f_rows = [{"Model": model_name, "Repeat": repeat, "Fold": fold, "Feature": f} for f in selected_features]
    return metrics, f_rows
excel_files = glob.glob(os.path.join(DATA_DIR, "*_model_table.xlsx"))
all_metrics = []
all_selected_features = []

for file_path in excel_files:
    model_name = os.path.basename(file_path).replace("_model_table.xlsx", "")
    print(f"Processing Pooled  Model: {model_name}")

    df = pd.read_excel(file_path, index_col=0)
    
    y = (df["disease"] == "T1D").astype(int)
    countries = df["country"]
    stratify_col = countries.astype(str) + "_" + y.astype(str)
    
    X_full = df.drop(["country", "dataset", "age", "sex", "disease"], axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_full = np.log1p(X_full)

    tasks = []
    for repeat in range(5):
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + repeat)
        for fold, (tr, ts) in enumerate(outer_cv.split(X_full, stratify_col)):
            tasks.append((model_name, repeat + 1, fold + 1, tr, ts))
    results = Parallel(n_jobs=-1)(delayed(run_fold)(*t, X_full, y, countries) for t in tasks)

    for res in results:
        if res:
            all_metrics.append(res[0])
            all_selected_features.extend(res[1])
pd.DataFrame(all_metrics).to_csv(OUTPUT_METRICS, index=False)
pd.DataFrame(all_selected_features).to_csv(OUTPUT_FEATURES, index=False)
print(f"\n Model run finished.")