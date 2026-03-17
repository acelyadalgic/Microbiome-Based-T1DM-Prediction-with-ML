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
from joblib import Parallel, delayed
import pyswarms as ps
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_DIR = "input_path"
OUTPUT_METRICS = "within_all_models_5x5_metrics_china_xgb.csv"
OUTPUT_FEATURES = "within_all_models_5x5_selected_features_china_xgb.csv"

excel_files = glob.glob(os.path.join(DATA_DIR, "*_model_table.xlsx"))
def pso_feature_selection(X_train, y_train, n_particles=30, iters=50):
    np.random.seed(RANDOM_STATE)
    n_features = X_train.shape[1]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def fitness_function(swarm_position):

        fitness_scores = []

        for particle in swarm_position:
            binary_mask = np.round(particle)
            selected_idx = np.where(binary_mask == 1)[0]

            if len(selected_idx) == 0:
                fitness_scores.append(1.0)
                continue

            aucs = []

            for tr_idx, val_idx in inner_cv.split(X_train, y_train):
                X_tr = X_train.iloc[tr_idx, selected_idx]
                X_val = X_train.iloc[val_idx, selected_idx]
                y_tr = y_train.iloc[tr_idx]
                y_val = y_train.iloc[val_idx]
                clf = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )

                clf.fit(X_tr, y_tr)
                y_val_prob = clf.predict_proba(X_val)[:, 1]
                aucs.append(roc_auc_score(y_val, y_val_prob))

            mean_auc = np.mean(aucs)
            alpha = 0.2
            feature_ratio = len(selected_idx) / n_features
            fitness = 1 - mean_auc + alpha * feature_ratio

            fitness_scores.append(fitness)

        return np.array(fitness_scores)

    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 2, 'p': 2}

    optimizer = ps.discrete.BinaryPSO(
        n_particles=n_particles,
        dimensions=n_features,
        options=options
    )

    cost, pos = optimizer.optimize(fitness_function, iters=iters, verbose=False)
    return np.where(np.round(pos) == 1)[0]
def run_fold(model_name, repeat, fold, train_idx, test_idx, X_full, y_full):

    print(f"[START] Model={model_name} | Repeat={repeat} | Fold={fold}", flush=True)

    X_train = X_full.iloc[train_idx].copy()
    X_test  = X_full.iloc[test_idx].copy()
    y_train = y_full.iloc[train_idx].copy()
    y_test  = y_full.iloc[test_idx].copy()
    prev = (X_train > 0).mean(axis=0)
    keep_prev = prev >= 0.10
    X_train = X_train.loc[:, keep_prev]
    X_test  = X_test.loc[:, keep_prev]
    var = X_train.var(axis=0)
    keep_var = var > 0.001
    X_train = X_train.loc[:, keep_var]
    X_test  = X_test.loc[:, keep_var]
    if X_train.shape[1] == 0:
        metrics = {
            "Model": model_name, "Repeat": repeat, "Fold": fold,
            "AUC": np.nan, "Accuracy": np.nan, "Precision": np.nan,
            "Recall": np.nan, "F1": np.nan, "MCC": np.nan
        }
        print(f"[DONE ] Model={model_name} | Repeat={repeat} | Fold={fold} (NO FEATURES)", flush=True)
        return metrics, []
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns,
        index=X_test.index
    )
    selected_idx = pso_feature_selection(X_train, y_train)

    if len(selected_idx) == 0:
        selected_features = list(X_train.columns)
    else:
        selected_features = list(X_train.columns[selected_idx])

    feature_rows = [{
        "Model": model_name,
        "Repeat": repeat,
        "Fold": fold,
        "Feature": f
    } for f in selected_features]
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    param_distributions = {
        "n_estimators": np.arange(100, 501, 100),
        "max_depth": np.arange(2, 9),
        "learning_rate": np.linspace(0.01, 0.3, 12),
        "subsample": np.linspace(0.6, 1.0, 5),
        "colsample_bytree": np.linspace(0.6, 1.0, 5),
        "min_child_weight": [1, 2, 5, 10],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 5, 10]
    }

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    )

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1  
    )

    random_search.fit(X_train[selected_features], y_train)
    best_model = random_search.best_estimator_
    y_prob = best_model.predict_proba(X_test[selected_features])[:, 1]
    y_pred = best_model.predict(X_test[selected_features])

    metrics = {
        "Model": model_name,
        "Repeat": repeat,
        "Fold": fold,
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    print(f"[DONE ] Model={model_name} | Repeat={repeat} | Fold={fold} | "
          f"AUC={metrics['AUC']:.3f} | nFeat={len(selected_features)}",
          flush=True)

    return metrics, feature_rows
all_metrics = []
all_selected_features = []

for file_path in excel_files:

    model_name = os.path.basename(file_path).replace("_model_table.xlsx", "")
    print(f"\n==============================")
    print(f"Processing: {model_name}")
    print(f"==============================")

    df = pd.read_excel(file_path, index_col=0)

    country = df["country"]
    y_raw = df["disease"]

    df_ml = df.drop(["country", "dataset", "age", "sex", "disease"], axis=1)

    X_full = df_ml.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_full = np.log1p(X_full)

    y_full = (y_raw == "T1D").astype(int)
    mask = country == "China"
    X_full = X_full[mask]
    y_full = y_full[mask]
    if y_full.nunique() < 2:
        print(f"⚠ Skipping {model_name}: only one class in China subset.")
        continue

    tasks = []
    for repeat in range(5):
        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE + repeat
        )

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full)):
            tasks.append((repeat + 1, fold + 1, train_idx, test_idx))

    results = Parallel(n_jobs=-1)(
        delayed(run_fold)(
            model_name,
            repeat,
            fold,
            train_idx,
            test_idx,
            X_full,
            y_full
        )
        for (repeat, fold, train_idx, test_idx) in tasks
    )

    for metrics, features in results:
        all_metrics.append(metrics)
        all_selected_features.extend(features)
metrics_df = pd.DataFrame(all_metrics)
features_df = pd.DataFrame(all_selected_features)

metrics_df.to_csv(OUTPUT_METRICS, index=False)
features_df.to_csv(OUTPUT_FEATURES, index=False)

print("Model run finished")
print("Saved metrics  ->", OUTPUT_METRICS)
print("Saved features ->", OUTPUT_FEATURES)
