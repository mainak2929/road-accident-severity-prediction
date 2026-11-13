# UK Road Collision Severity: Model Evaluation, Imbalance, Explainability & Limitations

**Author:** Mainak (Member 4: Evaluation, Imbalance, Explainability & Demo)
**Builds on:** Satyam's pipeline (`03_feature_engineering_and_ml_models.ipynb`, `models/`), Mayank's EDA, Shrinivas's cleaning.
**Source of every number below:** `04_model_evaluation_imbalance_explainability.ipynb` (to regenerate it, open `scripts/build_evaluation_notebook.ipynb`, set `EXECUTE = True` in the last cell, and click Run All). Tables are in `reports/evaluation/*.csv`, figures are `reports/figures/fig12`–`fig17`.

---

## 1. Bottom line

* **Selected model:** Random Forest with `class_weight="balanced_subsample"`, the existing `models/best_model_pipeline.joblib`. It was chosen by a rule fixed in advance, on a validation split taken from the training data. The holdout was not used for selection.
* **It is useful beyond accuracy, but only modestly.** Holdout macro F1 is **0.362**, against **0.291** for a model that always predicts "Slight". It catches **55.5 % of fatal collisions** (the majority-class model catches 0 %), and ranking collisions by its fatal score concentrates real fatalities: the **top 5 % hold 26 % of all fatal collisions (5.2× the base rate)**.
* **It is not reliable at the individual level.** Only **4.5 %** of "Fatal" predictions are actually fatal (about 21 false alarms per correct one). Macro ROC-AUC is **0.656**. Its "Fatal" signal amounts to *rural, high-speed, dark road*. It **misses fatal collisions in urban, ≤30 mph settings** (urban fatal recall is 20 %; at 20 mph it is 5 %).
* **No target leakage found.** No single input is suspiciously predictive (best single-field ROC-AUC is 0.589). Two **data-quality issues** affect interpretation (§4.3).

---

## 2. Agreed metrics and selection rule (fixed before results)

| Role | Metric |
|---|---|
| **Primary** | **Macro F1**: every class counts equally, so it cannot be gamed by predicting "Slight". |
| **Guard rails** | Fatal recall ≥ 0.50 **and** Serious recall ≥ 0.30. A candidate below either floor cannot be selected. |
| **Tie-break** (within 0.005 macro F1) | Fatal PR-AUC (threshold-free ranking quality). |
| Reported only | Accuracy, balanced accuracy, weighted F1, per-class precision/recall/F1, macro ROC-AUC (OvR), confusion matrix, lift, calibration. |

**Data roles:** selection used an inner **validation split** (20 % of the 415,992 training rows, stratified, seed 42). Final numbers come from the **untouched 103,999-row holdout**. **For the team to confirm:** the guard-rail floors are a proposal. Changing them means editing `GUARD_RAILS` in the notebook generator and re-running.

### Fairness checks (all passed)
* The 80/20 split is re-created exactly. The holdout rows are identical, in the same order, to `models/test_holdout_with_predictions.parquet`.
* Both saved pipelines reproduce their stored holdout predictions with **100 % agreement** (max probability difference 4e-16).
* Every experiment fits its preprocessing inside its own pipeline on training rows only. All candidates share the same features and base hyperparameters.
* Refitting the saved configurations under current library versions (scikit-learn 1.9.1, xgboost 3.4.1; the artifacts were trained with scikit-learn 1.7.2) agrees with the artifacts on 98.3 % (RF) and 95.6 % (XGB) of predictions. **The final numbers come from the saved artifact itself.**

---

## 3. Class-imbalance experiment

Resampling is applied **only inside `fit`, on training rows** (imblearn pipeline). Validation and holdout rows are never resampled. Plain SMOTE was not used because 15 of 25 inputs are categorical or binary, and interpolating one-hot columns creates impossible rows. **SMOTENC** (categorical-aware) was used instead.

**Validation split (used for selection):**

| Candidate | Macro F1 | Fatal recall | Serious recall | Fatal PR-AUC | Accuracy | Guard rails |
|---|---|---|---|---|---|---|
| Dummy (always Slight) | 0.291 | 0.000 | 0.000 | 0.015 | **0.775** | ✗ |
| RF, no handling | 0.293 | 0.000 | 0.004 | 0.060 | 0.775 | ✗ |
| **RF, class weights (selected)** | **0.366** | **0.564** | **0.319** | 0.061 | 0.560 | ✓ |
| RF, random undersampling | 0.339 | 0.668 | 0.242 | 0.059 | 0.519 | ✗ (serious) |
| RF, hybrid SMOTENC | **0.372** | 0.433 | 0.383 | 0.052 | 0.567 | ✗ (fatal) |
| XGB, no handling | 0.302 | 0.000 | 0.017 | 0.068 | 0.775 | ✗ |
| XGB, sample weights | 0.357 | 0.629 | 0.327 | 0.064 | 0.530 | ✓ |
| XGB, random undersampling | 0.340 | 0.660 | 0.293 | 0.064 | 0.505 | ✗ (serious) |
| *RF, class weights, no coordinates (ablation)* | 0.357 | 0.592 | 0.285 | 0.059 | 0.551 | n/a |

**Holdout (refit on full train, reported only):** the ranking is unchanged. Class-weighted RF scores 0.362, SMOTENC 0.372, weighted XGB 0.355, undersampled RF 0.342, unweighted RF 0.293. See `fig12_imbalance_experiments_holdout.png` and `imbalance_experiments_holdout.csv`.

**What this shows**
1. **Without imbalance handling, every model collapses to "Slight".** These models have the best accuracy (77.5 %) and zero fatal recall. Their *ranking* quality, however, is no worse (unweighted XGB has fatal PR-AUC 0.068). Imbalance handling mostly moves the **decision threshold**; it does not teach the model new information.
2. **Class weighting beats resampling here.** Random undersampling throws away about 98 % of the Slight rows, and serious recall falls below the floor. SMOTENC gives the best macro F1 because it trades fatal recall (0.43) for serious recall (0.38). It would be selected if the team cared less about missing fatal collisions.
3. **Why RF over XGB:** XGB-weighted has higher fatal recall (0.63) and balanced accuracy, but its macro F1 is 0.009 lower, outside the tie band. It also over-predicts Fatal more, which lowers fatal precision. If the team decides fatal recall matters most, XGB-weighted is the defensible alternative and is already saved.
4. **The coordinates ablation** costs only 0.009 macro F1, so the model is not mainly memorising locations.

---

## 4. Leakage audit

### 4.1 Timing / target encoding
* `number_of_casualties` (known only after the outcome), `collision_severity_label` (the target as text), IDs, raw date strings and `police_force` are all excluded (`leakage_column_inventory.csv`).
* `number_of_vehicles` and the scene conditions are **recorded at the scene, after the collision has happened**. That is legitimate for a *post-collision severity classifier* (report-time triage), but it means **this is not a pre-crash risk model**.
* `speed_limit` imputation in notebook 01 uses group medians computed on all rows before the split. It uses no target information and touched only a handful of rows, so the leakage is negligible but worth noting.

### 4.2 Empirical probe
A shallow, class-balanced tree was trained on **one field at a time** (`leakage_single_field_probe.csv`). The strongest single field is `number_of_vehicles` (macro ROC-AUC 0.589), followed by location (0.587) and speed limit (0.577). A leaked field would stand out far above the others, and none does. The excluded `number_of_casualties` turned out weak on its own (0.528), but it stays excluded on timing grounds.

### 4.3 Process artifacts and data-quality issues (action items for the team)
* **Junction label bug (Shrinivas/Satyam):** junction-detail codes 13, 16, 17, 18, 19, which are real junction types, are labelled `Missing / Unknown` in the cleaned data. So `junction_detail_label` really means only "at a junction vs not", and 48 % of rows lose their junction type. Fixing the code→label map in notebook 01 and retraining may help.
* **"Unknown" codes encode reporting, not road conditions.** Fatal rate is 0.04 % when road surface is "Unknown" and 0.18 % when road type is "Unknown", against 1.47 % overall. Collisions with incomplete records are almost never fatal (fatal collisions get fuller investigations). The model uses this: the most confident "missed serious" errors are records with Unknown road type and weather. It is a legitimate pattern in the data, but it is **not a physical risk factor** and would not carry over to a cleaner data feed.
* **Duplicated incidents across the split:** only 2 clusters with the same date, time and coordinates straddle train and test, so this is negligible.

---

## 5. Final evaluation on the holdout (selected model)

| | Precision | Recall | F1 | PR-AUC | Random PR-AUC | Support |
|---|---|---|---|---|---|---|
| **Fatal** | 0.045 | **0.555** | 0.083 | 0.056 | 0.015 | 1,529 |
| **Serious** | 0.272 | 0.322 | 0.295 | 0.243 | 0.210 | 21,874 |
| **Slight** | 0.839 | 0.614 | 0.709 | 0.853 | 0.775 | 80,596 |

Accuracy 0.552 · balanced accuracy 0.497 · **macro F1 0.362** · weighted F1 0.613 · macro ROC-AUC 0.656.

**Confusion matrix** (`fig13_confusion_matrix_selected_model.png`; rows = true class, columns = predicted)

| | pred Fatal | pred Serious | pred Slight |
|---|---|---|---|
| **true Fatal** | **849** | 400 | 280 |
| **true Serious** | 5,601 | **7,049** | 9,224 |
| **true Slight** | 12,582 | 18,497 | **49,517** |

**Lift: is the fatal score useful for ranking?** (`final_fatal_lift_table.csv`)

| Top % of collisions by fatal score | Fatal precision | Share of all fatal collisions captured | Lift |
|---|---|---|---|
| 1 % | 9.4 % | 6.4 % | 6.4× |
| 5 % | 7.6 % | 25.9 % | 5.2× |
| 10 % | 6.0 % | 40.8 % | 4.1× |
| 20 % | 4.3 % | 59.1 % | 3.0× |

**Calibration:** the scores are not probabilities. The mean "Fatal" score is 0.224, while the true fatal rate is 0.0147, an inflation of about **15×** caused by class weighting. The demo labels scores as such and shows a percentile instead.

---

## 6. Error analysis

**Which classes are confused** (`error_confusion_pairs.csv`)
1. **Slight → Serious (18,497)** and **Slight → Fatal (12,582)** are the largest error cells. This is the price of class weighting: **66 % of "Fatal" predictions are actually Slight** and 29 % are Serious.
2. **Serious → Slight (9,224, 42 % of all Serious)** is the biggest *miss*. Serious and Slight collisions share very similar recorded circumstances, and the injury outcome depends on things STATS19 collision rows lack (vulnerable road user, age, seatbelt, impact type).
3. Fatal collisions are rarely called Slight (18 %). When they are missed, they are more often called Serious (26 %).

**Where the model fails** (`error_segments.csv`)

| Segment | Fatal recall | % predicted Fatal | Accuracy |
|---|---|---|---|
| Rural | 0.78 | 49 % | 0.31 |
| Urban | **0.20** | 3 % | 0.67 |
| 20 mph | **0.05** | 0.5 % | 0.70 |
| 60 mph | 0.92 | **75 %** | **0.15** |
| Darkness, no lighting | 0.89 | 71 % | 0.16 |

The model reduces to a "rural / fast / unlit means fatal" rule. On 60 mph roads it calls 75 % of collisions Fatal, so accuracy there falls to 15 %.

**Caught vs missed fatal collisions** (`error_fatal_caught_vs_missed_profile.csv`)

| | Mean speed limit | % rural | % single vehicle | % ≤ 30 mph |
|---|---|---|---|---|
| Caught | 56 mph | 86 % | 45 % | 9 % |
| Missed → Serious | 35 mph | 36 % | **78 %** | 71 % |
| Missed → Slight | 36 mph | 21 % | 2 % | 72 % |

Missed fatal collisions look like ordinary urban collisions. The "single vehicle, urban, ≤30 mph" group is most likely **pedestrian fatalities**: in STATS19 a pedestrian is not a vehicle, so these rows show one vehicle. The model has no pedestrian or casualty-type feature and cannot see this. **This is the clearest gap to close next**, for example by joining casualty/vehicle tables for at-scene fields such as "pedestrian involved".

**Representative errors** are in `error_representative_examples.csv`: the 4 most confident errors of each of 4 types (missed fatal, false fatal alarm, missed serious, over-escalation). Two examples:
* The most confident *false fatal alarms* are all 60–70 mph, unlit, 01:00–03:00 collisions that turned out Slight. These circumstances really are high-risk, but most collisions in them are still Slight.
* The most confident *missed fatal* collisions are urban roundabout collisions at 30 mph in daylight, with nothing in the record that suggests severity.

---

## 7. Explainability: predictive importance is not causal influence

Three views (`fig10`, `fig15`, `fig16`, `fig17`):

* **Gini importance** (grouped): speed limit, latitude, longitude, kinetic-energy proxy, speed × rural, number of vehicles. Gini is biased towards continuous inputs, which inflates the coordinates.
* **Permutation importance on the holdout**, raw fields (shuffling `speed_limit` also changes its derived features):
  * by **Fatal PR-AUC drop**: speed limit (0.020) > number of vehicles (0.012) ≈ junction control (0.012) > road class (0.011) ≈ urban/rural (0.011) > hour (0.009) ≈ location (0.009).
  * by **macro F1 drop**: number of vehicles dominates (0.032). Shuffling speed limit or urban/rural slightly *raises* macro F1. Those fields drive the ranking but push too many collisions over the Fatal threshold, which confirms the over-flagging seen in §6.
* **SHAP** (TreeExplainer, 600 holdout rows balanced by class, one-hot summed): for **P(Fatal)**, the top contributors are speed limit, kinetic-energy proxy, speed × rural, junction control, urban/rural and number of vehicles/single-vehicle. Higher speed limits, rural areas, darkness and single-vehicle collisions push the Fatal score up.

**How to talk about this:** these numbers describe what the *model* relies on, given the correlations in police-reported data. They do **not** say that changing a speed limit, lighting or junction control would change real-world severity. Confounders are everywhere: rural roads also mean longer ambulance times and different road users, and "Unknown" codes reflect recording practice. Causal claims would need a causal design (e.g. before/after studies of speed-limit changes), not feature importance.

---

## 8. Limitations

1. **Weak discrimination.** Macro ROC-AUC is 0.66 and fatal precision is 4.5 %. The information that decides severity (who was hit, their age and protection, the impact type) is not in the collision-level features.
2. **Systematic blind spot:** urban, low-speed and probable pedestrian fatalities (fatal recall 5–20 % in these segments).
3. **Over-flagging** on rural, high-speed and unlit roads (up to 75 % of collisions there predicted Fatal).
4. **Scores are not probabilities**: class weighting inflates the fatal score about 15×. Calibrate (e.g. isotonic on a validation split) before showing any number as a "chance".
5. **Post-collision inputs.** This is a severity classifier at report time, not a pre-crash risk model.
6. **Reporting bias:** STATS19 covers only police-reported injury collisions. "Unknown" codes and location partly encode recording practice. Severity coding changed as forces moved to CRASH (the 2019–2023 window is mostly post-change, but adoption varied by force).
7. **Data-quality issues** (junction labels, §4.3) limit what junction features can contribute.
8. **Validation design:** the main split is random. A 2023 out-of-time check in notebook 03 shows stability, but there is no *spatial* holdout (unseen regions), so performance in new areas is untested.
9. **Library versions:** the artifacts were pickled with scikit-learn 1.7.2 and loading them in 1.9.1 gives version warnings. Predictions were verified identical, but production should pin versions or re-export.
10. **Hyperparameters** were not tuned. All candidates used notebook 03's settings, so the comparison is fair but not optimised.

---

## 9. Responsible-use note

* **Intended use:** education and analysis, i.e. exploring which recorded circumstances are associated with severe outcomes, and prioritising *aggregate* road-safety analysis.
* **Not intended for:** dispatching or triaging emergency response, insurance pricing, enforcement, judging individual drivers, or ranking specific locations as "dangerous" without further study.
* **Error costs are asymmetric and the model errs in both directions:** it misses 44 % of fatal collisions (mostly urban ones) and raises about 21 false alarms per correct fatal call. Any use must keep a human in the loop and be validated on the population where it would be used.
* **Fairness/geography:** the model performs very differently for urban and rural areas. Any aggregate use would under-represent urban, low-speed and pedestrian risk.
* **Communicate scores as relative ranks, not probabilities**, and always state that importance is associative.

---

## 10. Demo: Streamlit app

`streamlit run app/streamlit_app.py` from the project root.

* Loads the selected model named in `models/evaluation_summary.json` (fallback: `models/best_model_pipeline.joblib`) through `src/features.load_pipeline`. That helper fixes the pickling issue: the saved pipelines refer to a function defined in notebook 03's `__main__`, so they could not be loaded anywhere else.
* Inputs are exactly the fields the model uses. Dropdown options come from the fitted encoder's categories, and the time-of-day band is derived from the hour as in the cleaned data.
* Outputs: predicted class, per-class scores, a fatal-score percentile among holdout collisions, a "random holdout example" loader that shows the recorded severity, holdout metrics, and the responsible-use note.

### Final demo / testing checklist

**Pipeline and evaluation**
- [x] Saved pipelines load outside notebook 03 (`src.features.load_pipeline`)
- [x] Holdout split re-created; IDs identical to the exported holdout parquet
- [x] Saved artifacts reproduce stored holdout predictions (100 %)
- [x] Preprocessing fit on training rows only; resampling only inside `fit`
- [x] Metrics and selection rule fixed before results; selection on validation, not holdout
- [x] Confusion matrix, per-class P/R/F1, macro and weighted F1, accuracy reported
- [x] Imbalance comparison: none / class weights / undersampling / SMOTENC (RF and XGB)
- [x] Leakage audit: timing review, single-field probe, reporting artifacts, duplicate check, location ablation
- [x] Gini, permutation and SHAP importance, labelled as non-causal
- [x] Error analysis: confusion pairs, segment table, caught-vs-missed profile, representative errors
- [ ] **Team sign-off on the guard-rail floors** (Fatal ≥ 0.50, Serious ≥ 0.30)
- [ ] **Satyam/Shrinivas: confirm the junction-label fix** (and retrain if applied, then re-run notebook 04)

**Before the live demo**
- [ ] `pip install -r requirements.txt` in a clean environment; `python -c "from src.features import load_pipeline; load_pipeline('models/best_model_pipeline.joblib')"` succeeds
- [ ] `streamlit run app/streamlit_app.py` opens with no errors; the first load takes about 10 s (it scores the holdout for percentiles)
- [ ] Defaults (urban 30 mph, daytime, London) → expect **Slight** with a low fatal percentile
- [ ] Rural 60 mph, "Darkness - no lighting", 02:00, 1 vehicle → expect **Fatal** with a high percentile, then explain why this is still usually a false alarm (fatal precision 4.5 %)
- [ ] "Random fatal" a few times → show a missed urban fatal collision as a live failure case
- [ ] Have `fig13` (confusion matrix), `fig12` (imbalance comparison) and the lift table ready to answer "why not accuracy?"
- [ ] Read out the responsible-use note; do not present scores as probabilities or importance as causes
- [ ] Offline fallback: screenshots of the app and the executed notebook, in case of laptop or network issues
