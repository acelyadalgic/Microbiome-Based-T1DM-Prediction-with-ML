import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier 
import pyswarms as ps
import warnings

warnings.filterwarnings("ignore")
MODELS_DIRECTORY = "input_path" 
OUTPUT_DIRECTORY = "loso_xgb_china_to_italy"
TRAIN_COUNTRY = "China"
TEST_COUNTRY = "Italy"
RANDOM_STATE = 42

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

def pso_fitness(swarm_position, X_tr, y_tr):
    n_features = X_tr.shape[1]
    fitness_scores = []
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    alpha = 0.4
    
    for particle in swarm_position:
        binary_mask = np.round(particle)
        selected_idx = np.where(binary_mask == 1)[0]
        
        if len(selected_idx) == 0:
            fitness_scores.append(1.0) 
            continue
            
        aucs = []
        for tr_idx, val_idx in inner_cv.split(X_tr, y_tr):
            clf = XGBClassifier(
                n_estimators=50, 
                max_depth=5, 
                learning_rate=0.1,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                tree_method='hist' 
            )
            clf.fit(X_tr.iloc[tr_idx, selected_idx], y_tr.iloc[tr_idx])
            probs = clf.predict_proba(X_tr.iloc[val_idx, selected_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr.iloc[val_idx], probs))
        fitness = (1 - np.mean(aucs)) + alpha * (len(selected_idx) / n_features)
        fitness_scores.append(fitness)
    return np.array(fitness_scores)
all_files = glob.glob(os.path.join(MODELS_DIRECTORY, "*.xlsx"))

for file_path in all_files:
    file_name = os.path.basename(file_path).replace(".xlsx", "")
    print(f"\n[PROCESSING] {file_name}...")
    
    df = pd.read_excel(file_path, index_col=0)
    country = df["country"]
    y = (df["disease"] == "T1D").astype(int)
    X = df.drop(["country", "dataset", "age", "sex", "disease"], axis=1)
    X = np.log1p(X)
    X_train = X[country == TRAIN_COUNTRY]
    y_train = y[country == TRAIN_COUNTRY]
    X_test = X[country == TEST_COUNTRY]
    y_test = y[country == TEST_COUNTRY]
    keep_prev = (X_train > 0).mean(axis=0) >= 0.10
    X_train, X_test = X_train.loc[:, keep_prev], X_test.loc[:, keep_prev]
    
    keep_var = X_train.var(axis=0) > 0.001
    X_train, X_test = X_train.loc[:, keep_var], X_test.loc[:, keep_var]
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    print(f"PSO + XGBoost Feature Selection ({X_train_scaled.shape[1]} features)...")
    optimizer = ps.discrete.BinaryPSO(
        n_particles=30, 
        dimensions=X_train_scaled.shape[1], 
        options={'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 2, 'p': 2}
    )
    cost, pos = optimizer.optimize(pso_fitness, iters=50, X_tr=X_train_scaled, y_tr=y_train)
    
    selected_features = X_train_scaled.columns[np.where(np.round(pos) == 1)[0]].tolist()
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'scale_pos_weight': [1, ratio] 
    }
    
    grid = RandomizedSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, tree_method='hist'), 
        param_dist, 
        n_iter=15, 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    grid.fit(X_train_scaled[selected_features], y_train)
    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(X_test_scaled[selected_features])[:, 1]
    y_pred = best_model.predict(X_test_scaled[selected_features])
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pd.DataFrame(selected_features, columns=["Taxa"]).to_csv(f"{OUTPUT_DIRECTORY}/{file_name}_xgb_features.csv", index=False)
    summary = {
        "File": file_name, 
        "Train": TRAIN_COUNTRY, 
        "Test": TEST_COUNTRY,
        "Total_Features": len(selected_features), 
        "AUC": auc, 
        "Accuracy": acc,
        "F1_Score": f1,
        "Precision": prec,
        "Recall": rec,
        "MCC": mcc,
        "Internal_CV_AUC": grid.best_score_,
        "Best_Params": str(grid.best_params_)
      }
    
    output_file = f"{OUTPUT_DIRECTORY}/loso_xgb_summary.csv"
    pd.DataFrame([summary]).to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

print("Model run finished")
