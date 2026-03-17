import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from scikeras.wrappers import KerasClassifier
from joblib import Parallel, delayed
from tensorflow.keras.callbacks import EarlyStopping
import pyswarms as ps
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 


RANDOM_STATE = 42
DATA_DIR = "input_path"
OUTPUT_METRICS = "pooled_ann_pso_0.4_bn_es_final_metrics.csv"
OUTPUT_FEATURES = "pooled_ann_pso_0.4_bn_es_final_features.csv"

PSO_PARTICLES = 30 
PSO_ITERS = 50

def create_mlp_model(input_dim, units=32, dropout_rate=0.2, learning_rate=0.001):
    model = tf.keras.Sequential([
        
        tf.keras.layers.Dense(units, input_dim=input_dim, use_bias=False), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(dropout_rate),
        
        
        tf.keras.layers.Dense(units // 2, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=optimizer, 
        metrics=[ tf.keras.metrics.AUC(name='auc')]
    )
    return model

def pso_fitness_internal(swarm_position, X_train, y_train):
    n_features = X_train.shape[1]
    fitness_scores = []
    clf = LogisticRegression(penalty='l2', solver='liblinear', random_state=RANDOM_STATE)
    
    for particle in swarm_position:
        binary_mask = np.round(particle)
        selected_idx = np.where(binary_mask == 1)[0]
        
        if len(selected_idx) == 0:
            fitness_scores.append(1.0); continue

        clf.fit(X_train.iloc[:, selected_idx], y_train)
        probs = clf.predict_proba(X_train.iloc[:, selected_idx])[:, 1]
        auc = roc_auc_score(y_train, probs)
        
        fitness = (1 - auc) + 0.4 * (len(selected_idx) / n_features)
        fitness_scores.append(fitness)
        
    return np.array(fitness_scores)


def run_fold(model_name, repeat, fold, train_idx, test_idx, X_full, y_full, countries_full):
    tf.keras.backend.clear_session()
    
    X_tr_raw, X_te_raw = X_full.iloc[train_idx].copy(), X_full.iloc[test_idx].copy()
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
    c_train, c_test = countries_full.iloc[train_idx], countries_full.iloc[test_idx]

    keep_prev = (X_tr_raw > 0).mean(axis=0) >= 0.10
    X_tr_raw = X_tr_raw.loc[:, keep_prev]
    X_te_raw = X_te_raw.loc[:, keep_prev]

    keep_var = X_tr_raw.var(axis=0) > 0.001
    X_tr_raw = X_tr_raw.loc[:, keep_var]
    X_te_raw = X_te_raw.loc[:, keep_var]

    X_tr_sc = pd.DataFrame(index=X_tr_raw.index, columns=X_tr_raw.columns)
    X_te_sc = pd.DataFrame(index=X_te_raw.index, columns=X_te_raw.columns)

    for country in c_train.unique():
        m_tr = (c_train == country)
        m_te = (c_test == country)
        
        scaler = StandardScaler()
        X_tr_sc.loc[m_tr] = scaler.fit_transform(X_tr_raw.loc[m_tr])
        if m_te.any():
            X_te_sc.loc[m_te] = scaler.transform(X_te_raw.loc[m_te])

    X_tr_sc = X_tr_sc.apply(pd.to_numeric).fillna(0)
    X_te_sc = X_te_sc.apply(pd.to_numeric).fillna(0)

 
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 2, 'p': 2}
    optimizer = ps.discrete.BinaryPSO(n_particles=PSO_PARTICLES, dimensions=X_tr_sc.shape[1], options=options)
    
    cost, pos = optimizer.optimize(pso_fitness_internal, iters=PSO_ITERS, verbose=False, 
                                   X_train=X_tr_sc, y_train=y_train)
    
    selected_features = X_tr_sc.columns[np.where(np.round(pos) == 1)[0]].tolist()
    
    if not selected_features:
        return None

   
    early_stop = EarlyStopping(monitor='auc', mode='max', patience=15, restore_best_weights=True)

    final_clf = KerasClassifier(
        model=lambda: create_mlp_model(input_dim=len(selected_features), units=32),
        epochs=150, 
        batch_size=16, 
        verbose=0
    )
    final_clf.fit(X_tr_sc[selected_features], y_train, callbacks=[early_stop])

    y_prob = final_clf.predict_proba(X_te_sc[selected_features])[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "Model": model_name, "Repeat": repeat, "Fold": fold,
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Features_Selected": len(selected_features)
    }
    feature_rows = [{"Model": model_name, "Repeat": repeat, "Fold": fold, "Feature": f} for f in selected_features]
    return metrics, feature_rows


excel_files = glob.glob(os.path.join(DATA_DIR, "*_model_table.xlsx"))
all_metrics = []
all_selected_features = []

for file_path in excel_files:
    model_name = os.path.basename(file_path).replace("_model_table.xlsx", "")
    print(f"\n[START] Model: {model_name} | Particles: {PSO_PARTICLES} | Iters: {PSO_ITERS}")

    df = pd.read_excel(file_path, index_col=0)
    y = (df["disease"] == "T1D").astype(int)
    countries = df["country"]
    stratify_col = countries.astype(str) + "_" + y.astype(str)
    
    X_full = df.drop(["country", "dataset", "age", "sex", "disease"], axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_full = np.log1p(X_full)

    tasks = []
    for r in range(5):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
        for f, (tr, ts) in enumerate(skf.split(X_full, stratify_col)):
            tasks.append((model_name, r + 1, f + 1, tr, ts))


    results = Parallel(n_jobs=2)(delayed(run_fold)(*t, X_full, y, countries) for t in tasks)

    for res in results:
        if res:
            all_metrics.append(res[0])
            all_selected_features.extend(res[1])


pd.DataFrame(all_metrics).to_csv(OUTPUT_METRICS, index=False)
pd.DataFrame(all_selected_features).to_csv(OUTPUT_FEATURES, index=False)
print("Model run finished.")