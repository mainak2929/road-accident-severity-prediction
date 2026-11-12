import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04_model_evaluation_imbalance_explainability.ipynb"

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# 04 · Model Evaluation, Class Imbalance, Explainability & Error Analysis

**Owner:** Mainak (Member 4). **Inputs:** Satyam's saved pipelines in `models/` and the cleaned STATS19 2019–2023 parquet.

This notebook answers one question: **is the model useful beyond accuracy, and where does it fail?**

Protocol:
1. Fix the evaluation metrics and the selection rule **before** looking at any candidate's results.
2. Re-create Satyam's exact stratified 80/20 split and check the saved models reproduce their stored holdout predictions.
3. Audit for leakage.
4. Run the class-imbalance experiments. All candidates are compared on a **validation split carved out of the training set**, so the holdout plays no part in selection. Resampling happens only inside `fit`, on training rows.
5. Evaluate the selected model **once** on the untouched 103,999-row holdout: confusion matrix, per-class metrics, PR curves, lift, calibration.
6. Error analysis and explainability (Gini, permutation importance, SHAP).

All tables are written to `reports/evaluation/` and figures to `reports/figures/fig12+`.
""")

code(r"""import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, average_precision_score, roc_auc_score,
                             precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

from src.features import (extract_engineered_features, load_pipeline, load_data_and_split, time_of_day_band,
                          NUM_COLS, CAT_COLS, BIN_COLS, RAW_INPUT_COLS, CLASS_NAMES, RANDOM_STATE, MODELS_DIR)

pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)
EVAL_DIR = os.path.join("reports", "evaluation"); FIG_DIR = os.path.join("reports", "figures")
os.makedirs(EVAL_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

CLASS_COLORS = {"Fatal": "#2a78d6", "Serious": "#eb6834", "Slight": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})
def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight", facecolor="white")
print("ready")
""")

md("""
## 1 · Agreed evaluation metrics and selection rule (fixed before any results)

| Role | Metric | Why |
|---|---|---|
| **Primary (selection)** | **Macro F1** | Weights Fatal, Serious and Slight equally, so it cannot be gamed by predicting "Slight" (the dummy scores 77.5% accuracy but 0.29 macro F1). |
| Guard rail | Fatal recall ≥ 0.50, Serious recall ≥ 0.30 | Missing serious and fatal collisions is the costly error. A candidate below either floor cannot be selected, however high its macro F1. |
| Tie-break (within 0.005 macro F1) | Fatal PR-AUC (average precision) | Threshold-independent: how well the model *ranks* fatal cases. |
| Reported, not used for selection | Accuracy, balanced accuracy, weighted F1, per-class precision/recall/F1, macro ROC-AUC, confusion matrix | Full picture. Accuracy and weighted F1 are dominated by the 77.5% Slight class. |

The guard-rail floors are a proposal to the team, set slightly below what the balanced models in notebook 03 already reach. Changing them means changing `GUARD_RAILS` below and re-running the notebook.
""")

code(r"""PRIMARY = "Macro F1"
GUARD_RAILS = {"Fatal Recall": 0.50, "Serious Recall": 0.30}
TIE_BAND, TIE_BREAK = 0.005, "Fatal PR-AUC"

def evaluate(y_true, y_pred, proba=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    out = {"Accuracy": accuracy_score(y_true, y_pred),
           "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
           "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
           "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}
    for i, c in enumerate(CLASS_NAMES):
        out[f"{c} Precision"], out[f"{c} Recall"], out[f"{c} F1"] = p[i], r[i], f[i]
    if proba is not None:
        for i, c in enumerate(CLASS_NAMES):
            out[f"{c} PR-AUC"] = average_precision_score(y_true == i, proba[:, i])
        out["Macro ROC-AUC (OvR)"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    return out

def select(table):
    ok = table[table["candidate"]].copy()
    for m, floor in GUARD_RAILS.items():
        ok = ok[ok[m] >= floor]
    if ok.empty:
        raise RuntimeError("No candidate meets the guard rails")
    best = ok[PRIMARY].max()
    near = ok[ok[PRIMARY] >= best - TIE_BAND]
    return near.sort_values(TIE_BREAK, ascending=False).index[0], ok.index.tolist()
print("metric definitions locked")
""")

md("""
## 2 · Fair-comparison checks: same split, same preprocessing, reproducible artifacts
""")

code(r"""df, y, train_idx, test_idx = load_data_and_split()
df_train, df_test = df.loc[train_idx], df.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

inner_tr_idx, val_idx = train_test_split(train_idx, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE)
df_itr, df_val = df.loc[inner_tr_idx], df.loc[val_idx]
y_itr, y_val = y.loc[inner_tr_idx], y.loc[val_idx]

holdout = pd.read_parquet(os.path.join(MODELS_DIR, "test_holdout_with_predictions.parquet"))
checks = {
    "train/test disjoint": len(set(train_idx) & set(test_idx)) == 0,
    "inner-train/val disjoint and inside train": len(set(inner_tr_idx) & set(val_idx)) == 0 and set(val_idx) <= set(train_idx),
    "test ids identical to Satyam's holdout (same order)": bool((df_test["collision_index"].values == holdout["collision_index"].values).all()),
}
saved = {"RF balanced (saved artifact)": load_pipeline(os.path.join(MODELS_DIR, "best_model_pipeline.joblib")),
         "XGB weighted (saved artifact)": load_pipeline(os.path.join(MODELS_DIR, "xgboost_pipeline.joblib"))}
saved_proba = {k: m.predict_proba(df_test) for k, m in saved.items()}
checks["saved RF reproduces stored predictions"] = bool((saved_proba["RF balanced (saved artifact)"].argmax(1) == holdout["pred_rf_idx"].values).all())
checks["saved XGB reproduces stored predictions"] = bool((saved_proba["XGB weighted (saved artifact)"].argmax(1) == holdout["pred_xgb_idx"].values).all())
display(pd.Series(checks, name="passed").to_frame())
assert all(checks.values())

split_tbl = pd.DataFrame({name: yy.map(dict(enumerate(CLASS_NAMES))).value_counts(normalize=True).reindex(CLASS_NAMES)
                          for name, yy in [("inner-train", y_itr), ("validation", y_val), ("train (full)", y_train), ("holdout test", y_test)]}).T
split_tbl["N"] = [len(y_itr), len(y_val), len(y_train), len(y_test)]
display(split_tbl.style.format({c: "{:.2%}" for c in CLASS_NAMES} | {"N": "{:,}"}))
""")

md("""
## 3 · Leakage audit

Three questions:
1. **Timing.** Is any input only known *after* the collision outcome, or does it directly encode the target?
2. **Empirical power.** Does any single input predict severity suspiciously well on its own? A leaky field usually stands out. `number_of_casualties` is included as a deliberate positive control.
3. **Process artifacts.** Do missing/unknown codes, duplicated incidents or location memorisation inflate the scores?
""")

code(r"""used_raw = set(RAW_INPUT_COLS)
reasons = {
    "collision_severity": "TARGET", "collision_severity_label": "Target label (direct encoding)",
    "number_of_casualties": "Post-outcome: counted after injuries are known, strongly tied to severity",
    "collision_index": "Record ID", "date": "Raw string (month/weekend/hour derived instead)", "time": "Raw string",
    "date_parsed": "Raw datetime", "police_force": "Administrative reporting unit",
    "pedestrian_crossing": "Excluded by notebook 03 (operational)", "speed_limit_imputed": "Cleaning flag",
}
rows = []
for c in df.columns:
    if c in used_raw:
        status = "USED"
        note = {"number_of_vehicles": "Known at the scene; fine for report-time triage, NOT a pre-crash risk input",
                "latitude": "Location; risk of memorising local severity rates (ablated below)",
                "longitude": "Location; see latitude"}.get(c, "Pre-collision circumstance recorded at the scene")
    else:
        status = "excluded"
        note = reasons.get(c, "Not used by the pipeline (redundant code / label twin or not selected)")
    rows.append({"column": c, "status": status, "note": note})
inventory = pd.DataFrame(rows).sort_values(["status", "column"], ascending=[False, True])
inventory.to_csv(os.path.join(EVAL_DIR, "leakage_column_inventory.csv"), index=False)
display(inventory.reset_index(drop=True))
""")

code(r"""probe_fields = {f: [f] for f in RAW_INPUT_COLS if f not in ("latitude", "longitude")}
probe_fields["latitude+longitude"] = ["latitude", "longitude"]
probe_fields.update({"number_of_casualties (EXCLUDED, positive control)": ["number_of_casualties"],
                     "police_force (EXCLUDED)": ["police_force"],
                     "special_conditions_at_site (unused)": ["special_conditions_at_site"],
                     "carriageway_hazards (unused)": ["carriageway_hazards"]})
probe_rows = []
for name, cols in probe_fields.items():
    cat = [c for c in cols if df[c].dtype == object or str(df[c].dtype) in ("string", "str") or c == "police_force"]
    num = [c for c in cols if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)])
    m = Pipeline([("pre", pre), ("dt", DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, class_weight="balanced", random_state=RANDOM_STATE))])
    m.fit(df_itr[cols], y_itr)
    pr = m.predict_proba(df_val[cols])
    ev = evaluate(y_val, pr.argmax(1), pr)
    probe_rows.append({"field": name, "Macro F1": ev["Macro F1"], "Fatal Recall": ev["Fatal Recall"],
                       "Fatal PR-AUC": ev["Fatal PR-AUC"], "Macro ROC-AUC (OvR)": ev["Macro ROC-AUC (OvR)"]})
probe = pd.DataFrame(probe_rows).sort_values("Macro ROC-AUC (OvR)", ascending=False).reset_index(drop=True)
probe.to_csv(os.path.join(EVAL_DIR, "leakage_single_field_probe.csv"), index=False)
display(probe.style.format({c: "{:.3f}" for c in probe.columns if c != "field"}))
""")

code(r"""art_rows = []
for c in ["road_type_label", "weather_conditions_label", "road_surface_conditions_label", "light_conditions_label",
          "junction_detail_label", "junction_control_label"]:
    for lvl, g in df.groupby(c):
        if "unknown" in str(lvl).lower():
            art_rows.append({"field": c, "level": lvl, "N": len(g), "fatal rate": (g.collision_severity == 1).mean(),
                             "serious rate": (g.collision_severity == 2).mean()})
artifacts = pd.DataFrame(art_rows)
overall = (df.collision_severity == 1).mean()
print(f"Overall fatal rate: {overall:.2%}")
display(artifacts.style.format({"N": "{:,}", "fatal rate": "{:.2%}", "serious rate": "{:.2%}"}))

print("\nJunction label mapping check: raw junction_detail code vs label")
display(pd.crosstab(df["junction_detail"], df["junction_detail_label"]))
""")

code(r"""key = ["date", "time", "latitude", "longitude"]
part = pd.Series("train", index=df.index); part.loc[test_idx] = "test"
d2 = df[key + ["collision_severity"]].assign(part=part)
d2["cluster"] = d2.groupby(key, sort=False).ngroup()
grp = d2.groupby("cluster").agg(n=("part", "size"), parts=("part", "nunique"), sev=("collision_severity", "nunique"))
straddle = grp[grp.parts > 1]
n_test_rows_in_straddle = int(((d2.part == "test") & d2.cluster.isin(straddle.index)).sum())
print(f"Incident clusters with members in both train and test: {len(straddle):,}")
print(f"Test rows involved: {n_test_rows_in_straddle:,} ({n_test_rows_in_straddle/len(test_idx):.2%} of holdout)")
print(f"Of those clusters, share where every record has the same severity: {(straddle.sev == 1).mean():.1%}")
""")

md("""
## 4 · Class-imbalance experiments (selection on the validation split)

Every candidate uses the same feature engineering, the same `ColumnTransformer` (fit inside each pipeline, so on training rows only) and the same base hyperparameters as notebook 03.

| Candidate | Imbalance handling |
|---|---|
| Dummy (majority) | none |
| RF / XGB — none | none |
| RF — class weights | `class_weight="balanced_subsample"` (**= saved best_model_pipeline**) |
| XGB — sample weights | `compute_sample_weight("balanced")` (**= saved xgboost_pipeline**) |
| RF / XGB — random undersampling | `RandomUnderSampler` down to the Fatal count. Training-only, inside the imblearn pipeline |
| RF — hybrid SMOTENC | Undersample Slight to the Serious count, then **SMOTENC** synthesises Fatal up to the same count. Training-only. Plain SMOTE does not fit this data: 15 of 25 inputs are categorical or binary, and interpolating one-hot vectors produces impossible rows. SMOTENC handles categorical features properly. |
| RF — class weights, no coordinates | **Ablation, not a candidate.** Measures how much the model leans on location |

`imblearn.Pipeline` applies a sampler only during `fit`. At `predict` it is skipped, so validation and holdout rows are never resampled.
""")

code(r"""RF_PARAMS = dict(n_estimators=100, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE)
XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)

def fe():
    return FunctionTransformer(extract_engineered_features)

def onehot_pre(num_cols=NUM_COLS):
    return ColumnTransformer([("num", StandardScaler(), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
                              ("bin", "passthrough", BIN_COLS)])

n_num, n_cat = len(NUM_COLS), len(CAT_COLS)
cat_idx = list(range(n_num, n_num + n_cat)); bin_idx = list(range(n_num + n_cat, n_num + n_cat + len(BIN_COLS)))
def count_of(cls):
    return lambda yy: int(np.sum(np.asarray(yy) == cls))

def make(name):
    if name == "Dummy (majority)":
        return Pipeline([("fe", fe()), ("pre", onehot_pre()), ("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "RF - none":
        return Pipeline([("fe", fe()), ("pre", onehot_pre()), ("clf", RandomForestClassifier(**RF_PARAMS))])
    if name == "RF - class weights":
        return Pipeline([("fe", fe()), ("pre", onehot_pre()), ("clf", RandomForestClassifier(class_weight="balanced_subsample", **RF_PARAMS))])
    if name == "RF - class weights, no coordinates (ablation)":
        nc = [c for c in NUM_COLS if c not in ("latitude", "longitude")]
        return Pipeline([("fe", fe()), ("pre", onehot_pre(nc)), ("clf", RandomForestClassifier(class_weight="balanced_subsample", **RF_PARAMS))])
    if name == "RF - random undersampling":
        return ImbPipeline([("fe", fe()), ("pre", onehot_pre()), ("rus", RandomUnderSampler(random_state=RANDOM_STATE)),
                            ("clf", RandomForestClassifier(**RF_PARAMS))])
    if name == "RF - hybrid SMOTENC":
        ord_pre = ColumnTransformer([("num", StandardScaler(), NUM_COLS),
                                     ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
                                     ("bin", "passthrough", BIN_COLS)])
        post = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx)], remainder="passthrough")
        return ImbPipeline([("fe", fe()), ("pre", ord_pre),
                            ("rus", RandomUnderSampler(sampling_strategy=lambda yy: {2: count_of(1)(yy)}, random_state=RANDOM_STATE)),
                            ("smotenc", SMOTENC(categorical_features=cat_idx + bin_idx, sampling_strategy=lambda yy: {0: count_of(1)(yy)}, random_state=RANDOM_STATE)),
                            ("post", post), ("clf", RandomForestClassifier(**RF_PARAMS))])
    if name in ("XGB - none", "XGB - sample weights"):
        return Pipeline([("fe", fe()), ("pre", onehot_pre()), ("clf", XGBClassifier(**XGB_PARAMS))])
    if name == "XGB - random undersampling":
        return ImbPipeline([("fe", fe()), ("pre", onehot_pre()), ("rus", RandomUnderSampler(random_state=RANDOM_STATE)),
                            ("clf", XGBClassifier(**XGB_PARAMS))])
    raise KeyError(name)

EXPERIMENTS = ["Dummy (majority)", "RF - none", "RF - class weights", "RF - random undersampling", "RF - hybrid SMOTENC",
               "XGB - none", "XGB - sample weights", "XGB - random undersampling",
               "RF - class weights, no coordinates (ablation)"]
NOT_CANDIDATES = {"Dummy (majority)", "RF - class weights, no coordinates (ablation)"}

def fit_eval(name, X_fit, y_fit, X_eval, y_eval):
    m = make(name)
    kw = {"clf__sample_weight": compute_sample_weight("balanced", y_fit)} if name == "XGB - sample weights" else {}
    t = time.time(); m.fit(X_fit, y_fit, **kw); fit_s = time.time() - t
    pr = m.predict_proba(X_eval)
    return m, pr, {**evaluate(y_eval, pr.argmax(1), pr), "Fit time (s)": fit_s}

val_rows, val_proba = {}, {}
for name in EXPERIMENTS:
    _, pr, ev = fit_eval(name, df_itr, y_itr, df_val, y_val)
    val_rows[name], val_proba[name] = ev, pr
    print(f"{name:<50s} macroF1={ev['Macro F1']:.4f}  fatalR={ev['Fatal Recall']:.3f}  seriousR={ev['Serious Recall']:.3f}  ({ev['Fit time (s)']:.0f}s)")
val_tbl = pd.DataFrame(val_rows).T
val_tbl["candidate"] = [n not in NOT_CANDIDATES for n in val_tbl.index]
""")

code(r"""SHOW = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1", "Fatal Recall", "Serious Recall", "Slight Recall",
        "Fatal Precision", "Serious Precision", "Fatal PR-AUC", "Serious PR-AUC", "Macro ROC-AUC (OvR)", "Fit time (s)"]
chosen, eligible = select(val_tbl)
val_tbl["meets guard rails"] = val_tbl.index.isin(eligible)
val_tbl["selected"] = val_tbl.index == chosen
val_tbl.to_csv(os.path.join(EVAL_DIR, "imbalance_experiments_validation.csv"))
print(f"SELECTED (validation, by the rule fixed in section 1): {chosen}")
display(val_tbl[SHOW + ["meets guard rails", "selected"]].style.format({c: "{:.3f}" for c in SHOW}).format({"Fit time (s)": "{:.0f}"}))
""")

md("""
### 4b · Same experiments on the holdout (refit on the full training set, reported only)

Every candidate is refit on the full 415,992-row training set and scored on the same 103,999-row holdout. This table is **not** used for selection. It shows whether the validation ranking holds, and gives the numbers for the final write-up. The two saved artifacts are scored as well, to confirm the refits reproduce them.
""")

code(r"""hold_rows, hold_proba, hold_models = {}, {}, {}
for name in EXPERIMENTS:
    m, pr, ev = fit_eval(name, df_train, y_train, df_test, y_test)
    hold_rows[name], hold_proba[name], hold_models[name] = ev, pr, m
for name, pr in saved_proba.items():
    hold_rows[name] = {**evaluate(y_test, pr.argmax(1), pr), "Fit time (s)": np.nan}; hold_proba[name] = pr
hold_tbl = pd.DataFrame(hold_rows).T
hold_tbl.to_csv(os.path.join(EVAL_DIR, "imbalance_experiments_holdout.csv"))
for a, b in [("RF - class weights", "RF balanced (saved artifact)"), ("XGB - sample weights", "XGB weighted (saved artifact)")]:
    agree = (hold_proba[a].argmax(1) == hold_proba[b].argmax(1)).mean()
    print(f"refit '{a}' vs '{b}': prediction agreement {agree:.4%}")
display(hold_tbl[SHOW].style.format({c: "{:.3f}" for c in SHOW}).format({"Fit time (s)": "{:.0f}"}, na_rep="(saved)"))
""")

code(r"""order = [n for n in EXPERIMENTS]
panels = [("Macro F1", "Macro F1 (primary)"), ("Fatal Recall", "Fatal recall"), ("Serious Recall", "Serious recall"),
          ("Fatal PR-AUC", "Fatal PR-AUC (ranking)"), ("Accuracy", "Accuracy (misleading)")]
fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.6), sharey=True)
ypos = np.arange(len(order))
for ax, (col, title) in zip(axes, panels):
    vals = hold_tbl.loc[order, col].values
    colors = ["#2a78d6" if n == chosen else "#a9c6ec" for n in order]
    ax.barh(ypos, vals, color=colors, height=0.62)
    for yp, v in zip(ypos, vals):
        ax.text(v + 0.01, yp, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10); ax.set_xlim(0, max(1.0, vals.max() * 1.18) if col != "Fatal PR-AUC" else vals.max() * 1.35)
    ax.grid(axis="y", visible=False)
    if col in GUARD_RAILS:
        ax.axvline(GUARD_RAILS[col], color=INK2, lw=1, ls="--"); ax.text(GUARD_RAILS[col], len(order) - 0.4, " floor", fontsize=8, color=INK2)
axes[0].set_yticks(ypos); axes[0].set_yticklabels(order); axes[0].invert_yaxis()
fig.suptitle(f"Class-imbalance experiments on the 103,999-row holdout (dark bar = selected on validation: {chosen})", fontsize=11, fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); savefig("fig12_imbalance_experiments_holdout.png"); plt.show()
""")

md("""
## 5 · Final evaluation of the selected model on the holdout
""")

code(r"""SAVED_EQUIV = {"RF - class weights": ("RF balanced (saved artifact)", "best_model_pipeline.joblib"),
               "XGB - sample weights": ("XGB weighted (saved artifact)", "xgboost_pipeline.joblib")}
if chosen in SAVED_EQUIV:
    final_key, final_file = SAVED_EQUIV[chosen]
    final_model = saved[final_key]
    print(f"Selected config matches a saved artifact; evaluating models/{final_file}")
else:
    final_key, final_file = chosen, "selected_model_pipeline.joblib"
    final_model = hold_models[chosen]
    joblib.dump(final_model, os.path.join(MODELS_DIR, final_file))
    print(f"Selected config is new; saved refit pipeline to models/{final_file}")

proba = hold_proba[final_key]; pred = proba.argmax(1); yt = y_test.values
final_metrics = evaluate(yt, pred, proba)
print(classification_report(yt, pred, target_names=CLASS_NAMES, digits=3))
classwise = pd.DataFrame({c: {"Precision": final_metrics[f"{c} Precision"], "Recall": final_metrics[f"{c} Recall"],
                              "F1": final_metrics[f"{c} F1"], "PR-AUC": final_metrics[f"{c} PR-AUC"],
                              "Base rate (PR-AUC of random)": (yt == i).mean(), "Support": int((yt == i).sum())}
                          for i, c in enumerate(CLASS_NAMES)}).T
classwise.to_csv(os.path.join(EVAL_DIR, "final_classwise_metrics.csv"))
display(classwise.style.format("{:.3f}").format({"Support": "{:,.0f}"}))
summary = pd.Series({k: final_metrics[k] for k in ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC (OvR)"]})
display(summary.to_frame("holdout").style.format("{:.4f}"))
""")

code(r"""cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
pd.DataFrame(cm, index=[f"true {c}" for c in CLASS_NAMES], columns=[f"pred {c}" for c in CLASS_NAMES]).to_csv(os.path.join(EVAL_DIR, "final_confusion_matrix.csv"))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
views = [(cm, "Counts", "{:,.0f}"), (cm / cm.sum(1, keepdims=True), "Row-normalised = recall", "{:.1%}"),
         (cm / cm.sum(0, keepdims=True), "Column-normalised = precision", "{:.1%}")]
for ax, (mat, title, fmt) in zip(axes, views):
    norm = mat / mat.max()
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1.15)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10); ax.grid(False)
fig.suptitle(f"Confusion matrix, selected model ({final_key}), holdout N={len(yt):,}", fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); savefig("fig13_confusion_matrix_selected_model.png"); plt.show()
""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for i, (ax, c) in enumerate(zip(axes, CLASS_NAMES)):
    pr_, rc_, _ = precision_recall_curve(yt == i, proba[:, i])
    base = (yt == i).mean()
    ax.plot(rc_, pr_, color=CLASS_COLORS[c], lw=2)
    ax.axhline(base, color=INK2, ls="--", lw=1); ax.text(0.98, base, f"random = {base:.3f}", ha="right", va="bottom", fontsize=8, color=INK2)
    i_pt = final_metrics[f"{c} Recall"], final_metrics[f"{c} Precision"]
    ax.scatter(*i_pt, s=60, color=CLASS_COLORS[c], edgecolor="white", linewidth=2, zorder=3)
    ax.annotate("argmax decision", i_pt, xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_title(f"{c}: PR-AUC {final_metrics[f'{c} PR-AUC']:.3f}", fontsize=10)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout(); savefig("fig14_precision_recall_curves.png"); plt.show()
""")

code(r"""order_idx = np.argsort(-proba[:, 0]); base = (yt == 0).mean(); rows = []
for k in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
    top = order_idx[: int(round(k * len(yt)))]
    hits = (yt[top] == 0)
    rows.append({"top % by fatal score": f"{k:.1%}", "collisions flagged": len(top), "fatal captured": int(hits.sum()),
                 "precision": hits.mean(), "share of all fatals (recall)": hits.sum() / (yt == 0).sum(), "lift vs base rate": hits.mean() / base})
lift = pd.DataFrame(rows); lift.to_csv(os.path.join(EVAL_DIR, "final_fatal_lift_table.csv"), index=False)
print(f"Fatal base rate in holdout: {base:.2%}")
display(lift.style.format({"precision": "{:.2%}", "share of all fatals (recall)": "{:.1%}", "lift vs base rate": "{:.1f}x", "collisions flagged": "{:,}"}))

calib = pd.DataFrame({"mean predicted probability": proba.mean(0), "observed rate": [(yt == i).mean() for i in range(3)],
                      "Brier (one-vs-rest)": [np.mean((proba[:, i] - (yt == i)) ** 2) for i in range(3)]}, index=CLASS_NAMES)
calib["inflation factor"] = calib["mean predicted probability"] / calib["observed rate"]
calib.to_csv(os.path.join(EVAL_DIR, "final_probability_calibration.csv"))
display(calib.style.format("{:.4f}").format({"inflation factor": "{:.1f}x"}))
""")

md("""
## 6 · Error analysis
""")

code(r"""pairs = []
for i, t in enumerate(CLASS_NAMES):
    for j, p in enumerate(CLASS_NAMES):
        if i != j:
            pairs.append({"true": t, "predicted": p, "count": cm[i, j], "% of true class": cm[i, j] / cm[i].sum(),
                          "% of that prediction": cm[i, j] / cm[:, j].sum()})
pairs = pd.DataFrame(pairs).sort_values("count", ascending=False).reset_index(drop=True)
pairs.to_csv(os.path.join(EVAL_DIR, "error_confusion_pairs.csv"), index=False)
display(pairs.style.format({"count": "{:,}", "% of true class": "{:.1%}", "% of that prediction": "{:.1%}"}))
""")

code(r"""ev_df = df_test.assign(true=yt, pred=pred, p_fatal=proba[:, 0], p_serious=proba[:, 1], p_slight=proba[:, 2])
ev_df["vehicles"] = pd.cut(ev_df.number_of_vehicles, [0, 1, 2, 99], labels=["1 (single)", "2", "3+"])
ev_df["speed"] = ev_df.speed_limit.astype(int).astype(str) + " mph"

def seg(g):
    return pd.Series({"N": len(g), "fatal share": (g.true == 0).mean(),
                      "accuracy": accuracy_score(g.true, g.pred), "macro F1": f1_score(g.true, g.pred, average="macro", labels=[0, 1, 2], zero_division=0),
                      "fatal recall": ((g.true == 0) & (g.pred == 0)).sum() / max((g.true == 0).sum(), 1),
                      "serious recall": ((g.true == 1) & (g.pred == 1)).sum() / max((g.true == 1).sum(), 1),
                      "fatal precision": ((g.true == 0) & (g.pred == 0)).sum() / max((g.pred == 0).sum(), 1),
                      "% predicted Fatal": (g.pred == 0).mean()})
seg_tbls = []
for col in ["urban_or_rural_area_label", "speed", "vehicles", "light_conditions_label", "time_of_day_band", "road_surface_conditions_label"]:
    t = ev_df.groupby(col, observed=True).apply(seg)
    t = t[t.N >= 500]
    t.index = pd.MultiIndex.from_product([[col], t.index])
    seg_tbls.append(t)
segments = pd.concat(seg_tbls)
segments.to_csv(os.path.join(EVAL_DIR, "error_segments.csv"))
display(segments.style.format("{:.3f}").format({"N": "{:,.0f}", "fatal share": "{:.2%}", "% predicted Fatal": "{:.1%}"}))
""")

code(r"""fat = ev_df[ev_df.true == 0]
def profile(g):
    return pd.Series({"N": len(g), "mean speed limit": g.speed_limit.mean(), "% rural": (g.urban_or_rural_area_label == "Rural").mean(),
                      "% single vehicle": (g.number_of_vehicles == 1).mean(), "% daylight": (g.light_conditions_label == "Daylight").mean(),
                      "% night (19-07h)": g.time_of_day_band.str.startswith("Night").mean(), "% unlit dark": g.light_conditions_label.isin(["Darkness - no lighting", "Darkness - lights unlit"]).mean(),
                      "% 30 mph or less": (g.speed_limit <= 30).mean(), "median p_fatal": g.p_fatal.median()})
fatal_profile = pd.DataFrame({"caught (pred Fatal)": profile(fat[fat.pred == 0]),
                              "missed -> pred Serious": profile(fat[fat.pred == 1]),
                              "missed -> pred Slight": profile(fat[fat.pred == 2]),
                              "ALL non-fatal (reference)": profile(ev_df[ev_df.true != 0])}).T
fatal_profile.to_csv(os.path.join(EVAL_DIR, "error_fatal_caught_vs_missed_profile.csv"))
display(fatal_profile.style.format("{:.1%}").format({"N": "{:,.0f}", "mean speed limit": "{:.1f}", "median p_fatal": "{:.3f}"}))
""")

code(r"""SHOW_COLS = ["collision_index", "speed_limit", "urban_or_rural_area_label", "number_of_vehicles", "road_type_label",
             "light_conditions_label", "weather_conditions_label", "collision_hour", "junction_detail_label", "p_fatal", "p_serious", "p_slight"]
lab = dict(enumerate(CLASS_NAMES))
kinds = [("Missed fatal: true Fatal, predicted Slight", (ev_df.true == 0) & (ev_df.pred == 2), "p_slight"),
         ("False fatal alarm: true Slight, predicted Fatal", (ev_df.true == 2) & (ev_df.pred == 0), "p_fatal"),
         ("Missed serious: true Serious, predicted Slight", (ev_df.true == 1) & (ev_df.pred == 2), "p_slight"),
         ("Over-escalated: true Slight, predicted Serious", (ev_df.true == 2) & (ev_df.pred == 1), "p_serious")]
reps = []
for kind, mask, conf in kinds:
    r = ev_df[mask].sort_values(conf, ascending=False).head(4)[SHOW_COLS].assign(error_type=kind)
    reps.append(r)
reps = pd.concat(reps)[["error_type"] + SHOW_COLS].reset_index(drop=True)
reps.to_csv(os.path.join(EVAL_DIR, "error_representative_examples.csv"), index=False)
display(reps.style.format({"p_fatal": "{:.3f}", "p_serious": "{:.3f}", "p_slight": "{:.3f}", "speed_limit": "{:.0f}"}))
""")

md("""
## 7 · Explainability: what the model relies on

> **Predictive importance ≠ causal influence.** Every number below measures how much the *model's predictions* depend on an input, given the correlations in STATS19. None of it says that changing the input would change real-world severity. For example, `latitude/longitude` stand in for whatever varies by place: road design, traffic mix, distance to hospital, local police recording practice. A high-importance "Unknown" weather code reflects how collisions are *reported*, not the weather.

Three complementary views:
1. **Gini (mean decrease in impurity)** comes from the tree model. It is cheap but biased towards continuous, high-cardinality inputs such as coordinates.
2. **Permutation importance** shuffles each *raw input field* on a holdout sample and measures the drop in macro F1 and Fatal PR-AUC. It is model-agnostic and reflects holdout behaviour. Derived features move together, e.g. permuting `speed_limit` also changes `kinetic_energy_proxy`, `speed_x_rural` and `high_speed_unlit`.
3. **SHAP (TreeExplainer)** gives per-prediction contributions for each class. One-hot columns are summed back to their source field.
""")

code(r"""clf = final_model.steps[-1][1]
transform_part = final_model[:-1]
feat_names = None
for _, step in reversed(final_model.steps[:-1]):
    if isinstance(step, ColumnTransformer):
        try:
            feat_names = list(step.get_feature_names_out())
        except Exception:
            pass
        break
if feat_names is None or len(feat_names) != clf.n_features_in_:
    feat_names = [f"f{i}" for i in range(clf.n_features_in_)]
clean = [n.split("__", 1)[-1] for n in feat_names]

def raw_group(n):
    for c in CAT_COLS:
        if n.startswith(c + "_") or n == c:
            return c
    return n
if hasattr(clf, "feature_importances_"):
    gini = pd.DataFrame({"feature": clean, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    gini_grouped = gini.assign(group=gini.feature.map(raw_group)).groupby("group").importance.sum().sort_values(ascending=False)
    gini_grouped.to_csv(os.path.join(EVAL_DIR, "importance_gini_grouped.csv"))
    display(gini_grouped.head(15).to_frame("Gini importance (one-hot summed)"))
""")

code(r"""rng = np.random.default_rng(RANDOM_STATE)
samp_idx, _ = train_test_split(test_idx, train_size=20000, stratify=y_test, random_state=RANDOM_STATE)
Xs, ys = df.loc[samp_idx].copy(), y.loc[samp_idx].values
groups = {"speed_limit": ["speed_limit"], "number_of_vehicles": ["number_of_vehicles"],
          "location (lat+lon)": ["latitude", "longitude"], "hour (+ time_of_day_band)": ["collision_hour", "time_of_day_band"],
          "collision_month": ["collision_month"], "is_weekend": ["is_weekend"]}
groups.update({c: [c] for c in CAT_COLS if c != "time_of_day_band"})
def scores(X):
    p = final_model.predict_proba(X)
    return f1_score(ys, p.argmax(1), average="macro"), average_precision_score(ys == 0, p[:, 0])
base_f1, base_ap = scores(Xs)
perm_rows = []
for gname, cols in groups.items():
    d_f1, d_ap = [], []
    for rep in range(3):
        Xp = Xs.copy(); perm = rng.permutation(len(Xp))
        for c in cols:
            Xp[c] = Xs[c].values[perm]
        f1p, app = scores(Xp); d_f1.append(base_f1 - f1p); d_ap.append(base_ap - app)
    perm_rows.append({"input field": gname, "drop in macro F1": np.mean(d_f1), "sd (F1)": np.std(d_f1),
                      "drop in Fatal PR-AUC": np.mean(d_ap), "sd (AP)": np.std(d_ap)})
perm = pd.DataFrame(perm_rows).sort_values("drop in macro F1", ascending=False).reset_index(drop=True)
perm.to_csv(os.path.join(EVAL_DIR, "importance_permutation_raw_fields.csv"), index=False)
print(f"baseline on sample: macro F1 {base_f1:.4f}, Fatal PR-AUC {base_ap:.4f}")
display(perm.style.format({c: "{:.4f}" for c in perm.columns if c != "input field"}))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
yp = np.arange(len(perm))
for ax, (col, sd, title, color) in zip(axes, [("drop in macro F1", "sd (F1)", "Drop in macro F1 when shuffled", "#2a78d6"),
                                               ("drop in Fatal PR-AUC", "sd (AP)", "Drop in Fatal PR-AUC when shuffled", CLASS_COLORS["Fatal"])]):
    ax.barh(yp, perm[col], xerr=perm[sd], color=color, height=0.6, error_kw={"ecolor": INK2, "lw": 1})
    ax.axvline(0, color=INK2, lw=0.8); ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
axes[0].set_yticks(yp); axes[0].set_yticklabels(perm["input field"]); axes[0].invert_yaxis()
fig.suptitle("Permutation importance by raw input field: predictive reliance, NOT causal effect", fontweight="bold", x=0.01, ha="left")
plt.tight_layout(); savefig("fig15_permutation_importance_raw_fields.png"); plt.show()
""")

code(r"""SHAP_OK = False
try:
    import shap
    shap_idx = np.concatenate([rng.choice(np.where(y_test.values == k)[0], 200, replace=False) for k in range(3)])
    X_shap_raw = df_test.iloc[shap_idx]
    X_shap = transform_part.transform(X_shap_raw)
    cache = os.path.join(EVAL_DIR, f"shap_cache_{final_file.replace('.joblib', '')}.npz")
    ids = X_shap_raw["collision_index"].astype(str).values
    if os.path.exists(cache) and np.array_equal(np.load(cache, allow_pickle=True)["ids"], ids):
        sv = np.load(cache, allow_pickle=True)["sv"]
        print(f"SHAP values loaded from cache {cache} (delete it to recompute)")
    else:
        t = time.time()
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_shap, check_additivity=False)
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
        np.savez_compressed(cache, sv=sv, ids=ids)
        print(f"SHAP computed in {time.time() - t:.0f}s, shape {sv.shape}")
    SHAP_OK = True
except Exception as e:
    print("SHAP not available:", type(e).__name__, e)

if SHAP_OK:
    grp = pd.Series([raw_group(n) for n in clean])
    mean_abs = pd.DataFrame({c: pd.Series(np.abs(sv[:, :, i]).mean(0)).groupby(grp.values).sum() for i, c in enumerate(CLASS_NAMES)})
    mean_abs = mean_abs.loc[mean_abs.sum(1).sort_values(ascending=False).index]
    mean_abs.to_csv(os.path.join(EVAL_DIR, "importance_shap_mean_abs_grouped.csv"))
    top = mean_abs.head(14)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    yp = np.arange(len(top))
    for ax, c in zip(axes, CLASS_NAMES):
        ax.barh(yp, top[c], color=CLASS_COLORS[c], height=0.6); ax.set_title(f"mean |SHAP| for P({c})", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yp); axes[0].set_yticklabels(top.index); axes[0].invert_yaxis()
    fig.suptitle("SHAP importance per class (one-hot columns summed to source field)", fontweight="bold", x=0.01, ha="left")
    plt.tight_layout(); savefig("fig16_shap_importance_per_class.png"); plt.show()

    plt.figure()
    shap.summary_plot(sv[:, :, 0], X_shap, feature_names=clean, max_display=15, show=False, plot_size=(9, 6.5))
    plt.title("SHAP: direction of effect on the predicted FATAL score", fontsize=11, fontweight="bold")
    savefig("fig17_shap_beeswarm_fatal.png"); plt.show()
""")

md("""
## 8 · Summary for the write-up
""")

code(r"""summary_out = {
    "evaluated_on": f"stratified 20% holdout (random_state=42), N={len(yt)}; identical rows to models/test_holdout_with_predictions.parquet",
    "selection_rule": {"primary": PRIMARY, "guard_rails": GUARD_RAILS, "tie_break": TIE_BREAK, "tie_band": TIE_BAND,
                       "selection_data": "validation split (20% of training rows), never the holdout"},
    "selected_config": chosen, "selected_model_file": f"models/{final_file}",
    "validation_ranking": val_tbl.loc[val_tbl.candidate].sort_values(PRIMARY, ascending=False)[[PRIMARY, "Fatal Recall", "Serious Recall", "Fatal PR-AUC"]].round(4).to_dict(orient="index"),
    "holdout_metrics": {k: round(float(v), 4) for k, v in final_metrics.items()},
    "fatal_lift_top_5pct": float(lift.loc[lift["top % by fatal score"] == "5.0%", "lift vs base rate"].iloc[0]),
    "class_names": CLASS_NAMES,
}
with open(os.path.join(MODELS_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary_out, f, indent=2)
print(json.dumps(summary_out, indent=2))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

execute = "--execute" in sys.argv
if not execute and "--force" not in sys.argv and OUT.exists():
    existing = nbf.read(OUT, as_version=4)
    if any(c.get("outputs") for c in existing.cells if c.cell_type == "code"):
        sys.exit(f"{OUT.name} already has results. Use --execute to rebuild and re-run it, "
                 "or --force to overwrite it with a blank notebook.")

nbf.write(nb, OUT)
print("wrote", OUT)

if execute:
    from nbclient import NotebookClient

    try:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        print("executed OK")
    finally:
        nbf.write(nb, OUT)
