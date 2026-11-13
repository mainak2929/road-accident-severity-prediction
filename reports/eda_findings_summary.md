# UK Road Accident Severity Prediction — Comprehensive Exploratory Data Analysis & Statistical Storytelling Report

**Author / Primary Ownership**: **Mayank** (Member 2 — EDA, Statistics & Visual Storytelling Specialist)  
**Collaborators & Handoff Pipeline**: Shrinivas (Data Cleaning & Audit) $\rightarrow$ **Mayank (EDA)** $\rightarrow$ Satyam (Feature Engineering & ML Models) $\rightarrow$ Mainak (Evaluation & Explainability)  
**Dataset**: UK Department for Transport (DfT) STATS19 Road Safety Open Data (2019–2023 5-Year Cohort)  
**Sample Size**: $N = 519,991$ validated collisions | $47$ curated features  
**Associated Artifacts**:
* **Code-Only Clean Notebook**: [`02_exploratory_data_analysis.ipynb`](../02_exploratory_data_analysis.ipynb) (16 executable code cells, 8 multi-panel figures, zero markdown clutter)
* **High-Resolution Figures**: Saved at 300 DPI in [`reports/figures/`](figures/)

---

## 1. Executive Summary & The "Denominator Rule"
Exploratory data analysis of $519,991$ police-reported UK road collisions uncovers a critical, foundational insight: **collision exposure (where crashes happen most often) and collision severity (the intrinsic probability of severe injury or death) are governed by opposing mechanisms**.

Failing to normalize by the denominator produces the classic ecological fallacy of assuming that daytime urban driving is more hazardous than nighttime rural driving simply because urban daytime collisions exhibit the highest gross count. In reality:
$$\mathbb{P}(\text{Fatal} \mid \text{Circumstance}) = \frac{n_{\text{Fatal, Circumstance}}}{n_{\text{Total, Circumstance}}}$$

When conditional probabilities are computed, **fatality and serious injury rates are heavily concentrated in unlit rural environments, 60 mph undivided single carriageways, and off-peak early morning hours (01:00–04:00)**, despite these scenarios representing low gross traffic volume.

---

## 2. Tools, Environment & Technical Stack
The exploratory analysis pipeline was executed inside an isolated virtual environment (`.venv`) using the following scientific computing stack:
* **Python 3.12**: Core execution runtime.
* **Pandas ($\ge 2.0.0$)**: High-performance vectorized tabular aggregations, cross-tabulations, and Parquet I/O.
* **NumPy ($\ge 1.24.0$)**: Array mathematics, cumulative proportions, and multi-dimensional grid arrays.
* **SciPy (`scipy.stats` $\ge 1.10.0$)**: Parametric and non-parametric hypothesis testing ($\chi^2$ contingency tests, Cramér's $V$ with bias correction, Spearman rank correlation, skewness, and kurtosis).
* **Matplotlib ($\ge 3.7.0$) & Seaborn ($\ge 0.13.0$)**: Publication-grade statistical visualization (`seaborn-v0_8-whitegrid`, custom high-contrast colorblind-safe semantic palettes, dual-axis plotting, and 300 DPI vector-like export).
* **PyArrow & FastParquet**: Memory-mapped binary reading of Shrinivas's `cleaned_collisions_2019_2023.parquet` ($22.1\text{ MB}$, $0$ remaining null values).

---

## 3. Analytical & Statistical Methodologies

### A. The Denominator Rule (Dual-Representation Analysis)
Every situational variable was analyzed simultaneously across two dimensions:
1. **Gross Frequency (Exposure)**: Measures the raw volume of collisions to assess overall societal burden and frequency.
2. **Normalized Severity Proportions (Conditional Risk)**: Measures the conditional likelihood that a crash results in **Fatal**, **Serious**, or **Slight** outcomes.

### B. Statistical Significance & Association Metrics
* **Pearson's Chi-Square Contingency Test ($\chi^2$)**: Evaluates whether the distribution of accident severity is statistically independent of categorical environmental features (degrees of freedom: $\text{dof} = (r - 1)(c - 1)$).
* **Bias-Corrected Cramér's $V$**: Measures the nominal association strength between categorical predictors and the 3-class target:
  $$\tilde{\phi}^2 = \max\left(0, \frac{\chi^2}{n} - \frac{(k-1)(r-1)}{n-1}\right), \quad V = \sqrt{\frac{\tilde{\phi}^2}{\min(\tilde{r}-1, \tilde{k}-1)}}$$
* **Spearman's Rank Correlation ($\rho$)**: Evaluates monotonic relationships across ordinal and continuous predictors (`speed_limit`, `collision_hour`, `number_of_vehicles`, `urban_or_rural_area`, aligned ordinal `severity_rank`).
* **Tukey's Interquartile Range (IQR) Outlier Fences**: Identifies extreme values:
  $$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

### C. The Physical Hypothesis: Kinetic Energy Scaling
We tested Newtonian kinetic energy dissipation:
$$E_k = \frac{1}{2} m v^2$$
Hypothesizing that injury severity scales with the square of legal posted speed limits, amplified by single carriageway closing speeds during head-on impacts.

---

## 4. Key Findings & Empirical Evidence

### A. Target Variable Profiling & Temporal Stability
* **Severe Class Imbalance**:
  * **Slight**: $77.50\%$ ($n = 402,976$)
  * **Serious**: $21.03\%$ ($n = 109,369$)
  * **Fatal**: $1.47\%$ ($n = 7,646$)
  * *Imbalance Ratio*: **$53 : 14 : 1$**.
* **The Zero-Rule Fallacy**: A dummy classifier predicting always "Slight" achieves **$77.50\%$ accuracy** while scoring a fatal recall of $0.0\%$. **Accuracy is completely disqualified as an evaluation metric**.
* **COVID-19 Pandemic Impact (2020)**: Gross collision volume plummeted by **$-22.4\%$** in 2020 ($91,185$ vs. $117,508$ in 2019) due to travel restrictions. However, the **fatality rate rose** from $1.47\%$ to $1.60\%$ ($\chi^2 = 402.36, p = 5.8 \times 10^{-82}$). Open, deserted roads fostered extreme vehicle speeds, counteracting reduced traffic density. Proportions stabilized across 2021–2023 (Slight: $76.9\%-77.4\%$, Serious: $21.2\%-21.6\%$, Fatal: $1.4\%-1.5\%$).

### B. The "Commute Paradox" (Diurnal Cycle)
* **08:00 (Morning Peak)**: $32,927$ collisions (peak traffic), but only **$0.75\%$** fatal rate.
* **17:00 (Evening Peak)**: $45,133$ collisions (highest gross volume), but only **$1.05\%$** fatal rate.
* **03:00 (Early Morning Off-Peak)**: Only $3,524$ collisions (lowest volume), but **$4.26\%$ fatal rate ($5.7\times$ higher lethality)** and a $30.4\%$ combined severe (KSI) rate.
* *Mechanism*: Commute congestion causes low-speed bumper-to-bumper collisions; off-peak nighttime driving involves high cruising speeds, driver fatigue, reduced visibility, and alcohol/drug impairment.

### C. Kinetic Energy Scaling & The Rural Lethality Gap
* **Posted Speed Limit Escalation**: Fatality rates scale monotonically from **$0.43\%$** at 20 mph to **$0.86\%$** at 30 mph, **$1.77\%$** at 40 mph, **$2.92\%$** at 50 mph, and **$3.95\%$** at 60 mph ($9.2\times$ increase).
* **The 70 mph Anomaly**: 70 mph motorways exhibit a *lower* fatality rate ($3.20\%$) than 60 mph single carriageways ($3.95\%$) because motorways feature concrete median barriers, grade separation, no oncoming traffic, and zero pedestrian/cyclist conflict.
* **Urban vs. Rural Disparity ($3.46\times$)**: Rural collisions exhibit a **$2.84\%$ fatal rate** vs. **$0.82\%$ in urban areas** ($\chi^2 = 3568.1, \text{Cramér's } V = 0.083$). Rural roads feature higher cruising speeds, undivided lanes, and longer emergency medical service (EMS) dispatch times.

### D. Environmental Risk Factors (Cramér's $V$ Ranking)
* **Light Conditions ($\text{Cramér's } V = 0.088$)**: Darkness without street lights exhibits a **$4.90\%$ fatal rate** vs. **$1.20\%$ in daylight** ($4.1\times$ increase).
* **Weather Conditions ($\text{Cramér's } V = 0.038$)**: Fog and mist exhibit a **$3.52\%$ fatal rate** vs. $1.52\%$ in fine weather. High winds combined with rain increase fatality to $2.32\%$.
* **Road Surface ($\text{Cramér's } V = 0.041$)**: Flood water $> 3$cm deep has a $2.48\%$ fatal rate; wet/damp roads have $1.72\%$ fatal rate vs. $1.42\%$ on dry roads.
* **Road Type ($\text{Cramér's } V = 0.054$)**: Roundabouts are the safest intersection geometry ($0.44\%$ fatal) due to low-angle tangential merges. Single carriageways account for the vast majority of fatal impacts ($1.53\%$).

### E. Numerical Outliers & Multi-Vehicle Pileups
* **Pileup Validation**: Collisions involving $\ge 5$ vehicles ($n = 3,035$ incidents, $0.58\%$ of dataset) have a **$3.62\%$ fatal rate**, and $42.1\%$ occur on high-speed motorways. These represent genuine physical chain-reaction catastrophes, not data duplication.
* **Mass Casualties**: Collisions with $\ge 5$ casualties ($n = 2,842$) exhibit an elevated fatal rate of $4.71\%$.

---

## 5. The "Denominator Rule" Evidence Synthesis Table

| # | Specific Question / Claim | Observed Empirical Pattern | Sample Size ($N$) | Statistical Significance | Potential Confounders & Limitations |
|---|---|---|---|---|---|
| **1** | **COVID-19 Structural Break** | Gross volume fell $22.4\%$, but fatal rate rose from $1.47\%$ to $1.60\%$. | $N = 519,991$ ($91,185$ in 2020) | $\chi^2 = 402.36, p = 5.8 \times 10^{-82}$ | Empty roads encouraged higher speeding; travel restriction reporting bias. |
| **2** | **Light Conditions (Unlit Darkness)** | Darkness without street lighting exhibits $4.90\%$ fatal rate vs. $1.20\%$ in daylight ($4.1\times$). | $N = 27,876$ Unlit Darkness | $\text{Cramér's } V = 0.088, p < 10^{-300}$ | Unlit roads are predominantly rural single carriageways with 60 mph speed limits. |
| **3** | **Intersection Geometry & Roundabouts** | Roundabouts have only $0.44\%$ fatal rate vs. $1.53\%$ on single carriageways. | $N = 43,565$ Roundabouts | $\text{Cramér's } V = 0.054, p < 10^{-300}$ | Low impact angles and lower approach speeds minimize impact energy. |
| **4** | **Speed Limit Kinetic Scaling ($E_k \propto v^2$)** | Fatal rate rises monotonically from $0.43\%$ at 20 mph to $3.95\%$ at 60 mph ($9.2\times$). | $N = 519,991$ | Spearman $\rho = 0.142, p < 10^{-300}$ | 70 mph motorways have lower fatality ($3.20\%$) due to barrier protection and no pedestrian conflict. |
| **5** | **Urban vs. Rural Lethality Gap** | Rural roads have $2.84\%$ fatal rate vs. $0.82\%$ for urban roads ($3.46\times$). | $N = 168,095$ Rural vs. $351,896$ Urban | $\chi^2 = 3568.1, \text{Cramér's } V = 0.083$ | Confounded by speed limit and emergency medical transport distance. |
| **6** | **The Commute Paradox** | Morning rush (08:00) has $0.75\%$ fatal rate; 03:00 reaches $4.26\%$ ($5.7\times$). | $N = 32,927$ (8 AM) vs. $3,524$ (3 AM) | $\chi^2 = 1845.2, p < 10^{-300}$ | Congestion reduces kinetic speed; late-night drivers face fatigue and alcohol. |
| **7** | **Multi-Vehicle Pileups vs. Outliers** | Collisions involving $\ge 5$ vehicles exhibit $3.62\%$ fatal rate; $42\%$ occur on motorways. | $N = 3,035$ Pileups | $p < 10^{-50}$ | Genuine physical multi-impact chain reactions, not data duplication. |
| **8** | **Adverse Weather Risk** | Fog and mist have $3.52\%$ fatal rate vs. $1.52\%$ for fine weather ($2.3\times$). | $N = 4,203$ Fog collisions | $\chi^2 = 412.6, \text{Cramér's } V = 0.038$ | Low visibility reduces braking distance; low frequency creates small sample variance. |

---

## 6. Formal Handoff Manifesto for Satyam (ML) & Mainak (Evaluation)

### A. Candidate Feature Shortlist (Ranked by Predictive Power):
1. **`speed_limit`**: Primary physical predictor capturing impact kinetic energy.
2. **`urban_or_rural_area`**: Potent macro-environmental indicator for road design, lighting infrastructure, and emergency response latency.
3. **`light_conditions`**: High Cramér's V ($0.088$); unlit darkness strongly isolates fatal outcomes.
4. **`collision_hour` / `time_of_day_band`**: Captures diurnal behavioral shifts (The Commute Paradox).
5. **`road_type`**: Segregates high-protection roundabouts / dual carriageways from high-risk single carriageways.
6. **`first_road_class`**: Captures roadway hierarchy (Motorway, A-road, B-road, C-road, Unclassified).
7. **`road_surface_conditions`** & **`weather_conditions`**: Identifies loss-of-control risks (flooding, ice, fog/mist).
8. **`number_of_vehicles`**: Distinguishes single-vehicle run-off from multi-vehicle collisions.
9. **`is_weekend`**: Captures weekend leisure and late-night alcohol risk patterns.

### B. Critical Data Leakage Warning:
* ⚠️ **Strictly Exclude `number_of_casualties`**: In exploratory analysis, casualty count strongly correlates with severity, but casualties are an **after-the-fact outcome of the collision**, not an initial circumstance available at prediction time. Training a model with `number_of_casualties` will cause severe data leakage and render the model useless for real-time dispatch or pre-crash assessment.
* ⚠️ **Strictly Exclude Post-Crash Police Variables**: `police_force`, `pedestrian_crossing`, and site-clearing actions must not be used to predict collision severity.

### C. Class Imbalance Mitigation Strategy:
* **Train-Only Resampling Discipline**: If SMOTE, ADASYN, or Random Undersampling is tested, it **must only be applied within the training folds of cross-validation**. Resampling before the train/test split will leak validation distributions and invalidate results.
* **Algorithm-Native Cost Weighting**: Prioritize algorithm-native cost weighting (`class_weight='balanced'` in LightGBM/XGBoost/CatBoost) or custom focal loss over synthetic oversampling.
* **Cost-Sensitive Decision Thresholds**: Mainak must evaluate predictions using an asymmetric cost matrix penalizing Fatal false negatives at least $15\times$ to $20\times$ heavier than Slight false positives.
* **Evaluation Metrics**: Report Macro-F1, per-class PR-AUC, and Confusion Matrix with class-wise recall. Accuracy is prohibited.

### D. Dataset Scope Limitations (What the Data Cannot Answer):
* **Vehicle Speed at Impact**: The dataset records only the legal posted speed limit, not the telemetry speed at impact.
* **Toxicology & Impairment**: Breathalyzer/BAC readings are omitted from open STATS19 releases.
* **Driver Distraction / Mobile Phone Usage**: Not recorded in standard open collision tables.
* **Near-Misses**: Unreported near-collisions are completely absent from police administrative records.

---

## 7. Artifacts & Deliverables Registry
* **Executable Jupyter Notebook**: [`02_exploratory_data_analysis.ipynb`](../02_exploratory_data_analysis.ipynb)  
  *(16 clean code cells, zero markdown cells, fully executed with outputs, tables, and 8 inline plots).*
* **High-Resolution Figures (300 DPI PNG)**:
  * [`fig1_target_distribution_and_annual_stability.png`](figures/fig1_target_distribution_and_annual_stability.png)
  * [`fig2_univariate_circumstance_landscape.png`](figures/fig2_univariate_circumstance_landscape.png)
  * [`fig3_environmental_bivariate_risk.png`](figures/fig3_environmental_bivariate_risk.png)
  * [`fig4_speed_limit_urban_rural.png`](figures/fig4_speed_limit_urban_rural.png)
  * [`fig5_diurnal_commute_paradox.png`](figures/fig5_diurnal_commute_paradox.png)
  * [`fig6_numerical_distributions_outliers.png`](figures/fig6_numerical_distributions_outliers.png)
  * [`fig7_correlation_matrix.png`](figures/fig7_correlation_matrix.png)
  * [`fig8_spatial_collision_density.png`](figures/fig8_spatial_collision_density.png)
