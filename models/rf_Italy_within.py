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
import pyswarms as ps
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_DIR = "input_path"
OUTPUT_METRICS = "within_all_models_5x5_metrics_rf_italy.csv"
OUTPUT_FEATURES = "within_all_models_5x5_selected_features_rf_italy.csv"

excel_files = glob.glob(os.path.join(DATA_DIR, "*_model_table.xlsx"))

all_metrics = []
all_selected_features = []
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

            for train_idx, val_idx in inner_cv.split(X_train, y_train):

                X_tr = X_train.iloc[train_idx, selected_idx]
                X_val = X_train.iloc[val_idx, selected_idx]
                y_tr = y_train.iloc[train_idx]
                y_val = y_train.iloc[val_idx]

                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample"
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

    options = {'c1':0.5, 'c2':0.5, 'w':0.9, 'k':2, 'p':2}

    optimizer = ps.discrete.BinaryPSO(
        n_particles=n_particles,
        dimensions=n_features,
        options=options
    )

    cost, pos = optimizer.optimize(fitness_function, iters=iters, verbose=False)
    selected_indices = np.where(np.round(pos) == 1)[0]

    return selected_indices
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
    mask = country == "Italy"
    X_full = X_full[mask]
    y_full = y_full[mask]
    for repeat in range(5):

        print(f"\n----- REPEAT {repeat+1}/5 -----")

        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE + repeat
        )

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full)):

            print(f"Repeat {repeat+1} | Fold {fold+1}")

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
            selected_features = list(X_train.columns[selected_idx])

            for feat in selected_features:
                all_selected_features.append({
                    "Model": model_name,
                    "Repeat": repeat+1,
                    "Fold": fold+1,
                    "Feature": feat
                })
            param_distributions = {
                'n_estimators': np.arange(100,301,100),
                'max_depth':[None]+list(np.arange(5,31,5)),
                'min_samples_split':np.arange(2,11,2),
                'min_samples_leaf':np.arange(1,11,2),
                'max_features':['sqrt','log2',None],
                'bootstrap':[True,False]
            }

            rf = RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample"
            )

            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_distributions,
                n_iter=10,
                cv=3,
                scoring='roc_auc',
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            random_search.fit(X_train[selected_features], y_train)
            best_model = random_search.best_estimator_

            y_pred = best_model.predict(X_test[selected_features])
            y_prob = best_model.predict_proba(X_test[selected_features])[:,1]

            all_metrics.append({
                "Model": model_name,
                "Repeat": repeat+1,
                "Fold": fold+1,
                "AUC": roc_auc_score(y_test, y_prob),
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            })
metrics_df = pd.DataFrame(all_metrics)
features_df = pd.DataFrame(all_selected_features)

metrics_df.to_csv(OUTPUT_METRICS, index=False)
features_df.to_csv(OUTPUT_FEATURES, index=False)

print("\n Model run finished.")

