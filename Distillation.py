# Distillation_fixed_stockformer_metrics_dual.py
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib
from datetime import datetime, timezone

# -------- utilities --------
def log_string(logfile, s):
    t = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t} UTC] {s}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12)))

def accuracy(y_true, y_pred):
    # Sign for regression, threshold 0.5 for classification
    if set(np.unique(y_true)) <= {0, 1}:
        return np.mean((y_pred >= 0.5) == (y_true == 1))
    else:
        return np.mean(np.sign(y_true) == np.sign(y_pred))

# -------- parser --------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default=r"C:\Users\ns243\Documents\Academic\AI Master\Internship\Data\final_distillation_data_dual.csv",
                    help="Merged dataset with regression and classification targets")
parser.add_argument("--out_dir", type=str,
                    default=r"C:\Users\ns243\Documents\Academic\AI Master\Internship\Data\distill_output_dual",
                    help="Directory to save models, metrics, predictions")
parser.add_argument("--train_ratio", type=float, default=0.70)
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument("--test_ratio", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lgb_rounds", type=int, default=1000)
parser.add_argument("--early_stopping", type=int, default=50)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
log_file = os.path.join(args.out_dir, "distill_train_dual.log")
with open(log_file, "w") as f:
    f.write("")

log_string(log_file, "Starting dual-task distillation (regression + classification)")

# -------- load data --------
log_string(log_file, f"Loading data from: {args.data_path}")
df = pd.read_csv(args.data_path)

required_cols = [
    "Y_true_regression", "teacher_regression",
    "Y_true_classification", "teacher_classification"
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
if "instrument" not in df.columns:
    if "stock" in df.columns:
        df.rename(columns={"stock": "instrument"}, inplace=True)
    else:
        raise ValueError("Missing instrument/stock column")

df = df.dropna(subset=required_cols + ["date", "instrument"]).reset_index(drop=True)
df["instrument"] = df["instrument"].astype(str)

before = len(df)
df = df.drop_duplicates(subset=["date", "instrument"], keep="last").reset_index(drop=True)
after = len(df)
if before != after:
    log_string(log_file, f"Removed {before - after} duplicate rows")

log_string(log_file, f"Pre-split diagnostics: rows={len(df)}, dates={df['date'].nunique()}, instruments={df['instrument'].nunique()}")

# -------- feature detection --------
non_feature_cols = set(required_cols + ["date", "datetime", "instrument", "stock", "window"])
feature_cols = [c for c in df.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])]
feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
log_string(log_file, f"Using {len(feature_cols)} numeric feature columns")

# -------- chronological split --------
df = df.sort_values("date").reset_index(drop=True)
unique_dates = df["date"].drop_duplicates().sort_values()
n_dates = len(unique_dates)
train_end = int(n_dates * args.train_ratio)
val_end = train_end + int(n_dates * args.val_ratio)

train_dates = unique_dates.iloc[:train_end]
val_dates = unique_dates.iloc[train_end:val_end]
test_dates = unique_dates.iloc[val_end:]

train_df = df[df["date"].isin(train_dates)]
val_df = df[df["date"].isin(val_dates)]
test_df = df[df["date"].isin(test_dates)]

X_train, X_val, X_test = [d[feature_cols].fillna(0.0) for d in (train_df, val_df, test_df)]

def train_and_evaluate(task_name, y_true_col, y_teacher_col):
    log_string(log_file, f"\n=== {task_name.upper()} TASK ===")

    ytrain_true, yval_true, ytest_true = [d[y_true_col].values for d in (train_df, val_df, test_df)]
    ytrain_teacher, yval_teacher, ytest_teacher = [d[y_teacher_col].values for d in (train_df, val_df, test_df)]

    if task_name == "classification":
        params = dict(
            objective="binary",
            metric="binary_error",
            verbosity=-1,
            boosting_type="gbdt",
            seed=args.seed,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
        )
    else:
        params = dict(
            objective="regression",
            metric="rmse",
            verbosity=-1,
            boosting_type="gbdt",
            seed=args.seed,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
        )

    lgb_train_true = lgb.Dataset(X_train, label=ytrain_true)
    lgb_val_true = lgb.Dataset(X_val, label=yval_true, reference=lgb_train_true)
    lgb_train_teacher = lgb.Dataset(X_train, label=ytrain_teacher)
    lgb_val_teacher = lgb.Dataset(X_val, label=yval_teacher, reference=lgb_train_teacher)

    # Train baseline
    log_string(log_file, f"Training baseline LightGBM on true {task_name} labels")
    bst_true = lgb.train(params, lgb_train_true,
                         num_boost_round=args.lgb_rounds,
                         valid_sets=[lgb_train_true, lgb_val_true],
                         valid_names=["train", "val"],
                         callbacks=[lgb.early_stopping(args.early_stopping),
                                    lgb.log_evaluation(50)])
    bst_true.save_model(os.path.join(args.out_dir, f"lgb_baseline_true_{task_name}.txt"))

    # Train distilled
    log_string(log_file, f"Training distilled LightGBM on teacher {task_name} labels")
    bst_distill = lgb.train(params, lgb_train_teacher,
                            num_boost_round=args.lgb_rounds,
                            valid_sets=[lgb_train_teacher, lgb_val_teacher],
                            valid_names=["train", "val"],
                            callbacks=[lgb.early_stopping(args.early_stopping),
                                       lgb.log_evaluation(50)])
    bst_distill.save_model(os.path.join(args.out_dir, f"lgb_distilled_teacher_{task_name}.txt"))

    # Predict
    pred_baseline = bst_true.predict(X_test, num_iteration=bst_true.best_iteration)
    pred_distilled = bst_distill.predict(X_test, num_iteration=bst_distill.best_iteration)
    pred_teacher = ytest_teacher

    if task_name == "classification":
        pred_baseline = np.clip(pred_baseline, 0, 1)
        pred_distilled = np.clip(pred_distilled, 0, 1)
        pred_teacher = np.clip(pred_teacher, 0, 1)

    # Metrics
    results = {}
    for name, preds in [("teacher", pred_teacher),
                        ("lgb_baseline", pred_baseline),
                        ("lgb_distilled", pred_distilled)]:
        acc = accuracy(ytest_true, preds)
        mae = mean_absolute_error(ytest_true, preds)
        rrmse = rmse(ytest_true, preds)
        mape_val = mape(ytest_true, preds)
        results[name] = {"ACC": acc, "MAE": mae, "RMSE": rrmse, "MAPE": mape_val}
        log_string(log_file, f"{task_name}-{name} | acc={acc:.4f}, mae={mae:.4f}, rmse={rrmse:.4f}, mape={mape_val:.4f}")

    # Save predictions
    preds_out = test_df[["date", "instrument"]].copy()
    preds_out[f"Y_true_{task_name}"] = ytest_true
    preds_out[f"teacher_{task_name}"] = pred_teacher
    preds_out[f"lgb_baseline_{task_name}"] = pred_baseline
    preds_out[f"lgb_distilled_{task_name}"] = pred_distilled
    preds_out.to_csv(os.path.join(args.out_dir, f"test_predictions_{task_name}.csv"), index=False)

    joblib.dump(bst_true, os.path.join(args.out_dir, f"lgb_baseline_true_{task_name}.pkl"))
    joblib.dump(bst_distill, os.path.join(args.out_dir, f"lgb_distilled_teacher_{task_name}.pkl"))

    return results

# Run regression + classification
metrics_reg = train_and_evaluate("regression", "Y_true_regression", "teacher_regression")
metrics_cls = train_and_evaluate("classification", "Y_true_classification", "teacher_classification")

# Save all metrics
all_metrics = {"regression": metrics_reg, "classification": metrics_cls}
with open(os.path.join(args.out_dir, "metrics_stockformer_dual.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

log_string(log_file, "\n=== Final Evaluation Summary (Stockformer-style) ===")
for task_name, task_metrics in all_metrics.items():
    log_string(log_file, f"\n--- {task_name.upper()} ---")
    for model, vals in task_metrics.items():
        log_string(log_file, f"{model:>15s} | acc={vals['ACC']:.4f}, mae={vals['MAE']:.4f}, rmse={vals['RMSE']:.4f}, mape={vals['MAPE']:.4f}")

log_string(log_file, f"\nMetrics saved to: {os.path.join(args.out_dir, 'metrics_stockformer_dual.json')}")
log_string(log_file, "Done.")


