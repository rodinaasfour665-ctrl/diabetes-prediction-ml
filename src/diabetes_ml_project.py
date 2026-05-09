
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")          # keep output clean

import joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.show()

from scipy          import stats
from scipy.stats    import skew, kurtosis

from sklearn.preprocessing      import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.svm                import SVC
from sklearn.metrics            import (accuracy_score, f1_score,
                                        roc_auc_score, classification_report,
                                        confusion_matrix, ConfusionMatrixDisplay)
from sklearn.cluster            import KMeans
from sklearn.metrics            import silhouette_score, davies_bouldin_score
from sklearn.decomposition      import PCA
from sklearn.linear_model import LinearRegression as _LR

def variance_inflation_factor(X, idx):
    """Manual VIF: regress col idx on all others, return 1/(1-R^2)."""
    y_vif = X[:, idx]
    X_vif = np.delete(X, idx, axis=1)
    r2 = _LR().fit(X_vif, y_vif).score(X_vif, y_vif)
    return 1.0 / (1.0 - r2) if r2 < 1 else np.inf
from sklearn.utils              import resample

# ──────────────────────────────────────────────────────────────────────
# HELPER  – saves every figure to a PNG file
# ──────────────────────────────────────────────────────────────────────
FIGURE_DIR = "./outputs/"
fig_counter = [0]

def savefig(name):
    fig_counter[0] += 1
    path = f"{FIGURE_DIR}{fig_counter[0]:02d}_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  → saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  PART 1 – DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  PART 1 – DATA PREPARATION")
print("="*70)


# ──────────────────────────────────────────────────────────────────────
# 1.1  DATA UNDERSTANDING
# ──────────────────────────────────────────────────────────────────────


print("\n── 1.1  DATA UNDERSTANDING ──")

# ----- Load -----
df = pd.read_csv("data/diabetes.csv")
print("\n[READ] Dataset loaded successfully.")

# ----- Random samples -----
print("\n[RANDOM SAMPLES] — shows 5 random rows so we see varied data:")
print(df.sample(5, random_state=42).to_string())

# ----- Shape -----
print(f"\n[SHAPE] rows={df.shape[0]}  columns={df.shape[1]}")
print("  → 768 patients, 9 attributes (8 features + 1 target)")

# ----- Column names -----
print("\n[COLUMNS]:", df.columns.tolist())

# ----- Info -----
print("\n[INFO] — dtypes and non-null counts:")
df.info()

# ----- Describe -----
print("\n[DESCRIBE] — statistical summary:")
print(df.describe().round(2).to_string())

# ----- Unique values per column -----
print("\n[UNIQUE VALUES PER COLUMN]:")
for col in df.columns:
    print(f"  {col:30s}: {df[col].nunique()} unique  |  "
          f"sample values: {df[col].unique()[:5]}")

# ----- Class balance -----
print("\n[CLASS BALANCE]  (Outcome: 0=No Diabetes, 1=Diabetes)")
balance = df["Outcome"].value_counts()
print(balance)
ratio = balance[0] / balance[1]
print(f"  → Class ratio 0:1 = {ratio:.2f}  "
      f"({'moderately imbalanced' if ratio > 1.5 else 'balanced'})")

# ── Plot class balance ──
fig, ax = plt.subplots(figsize=(5, 4))
balance.plot(kind="bar", color=["#4C72B0", "#DD8452"], ax=ax, edgecolor="k")
ax.set_title("Class Balance (0=No Diabetes, 1=Diabetes)", fontsize=13)
ax.set_xlabel("Outcome"); ax.set_ylabel("Count")
ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"], rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x()+0.4, p.get_height()+5), ha="center")
plt.tight_layout()
savefig("class_balance")


# ──────────────────────────────────────────────────────────────────────
# 1.2  DATA CLEANING
# ──────────────────────────────────────────────────────────────────────


print("\n\n── 1.2  DATA CLEANING ──")

# ----- 1.2a  Duplicates -----

dups = df.duplicated().sum()
print(f"\n[DUPLICATES] Found: {dups}")
df = df.drop_duplicates()
print(f"  → After removal: {df.shape[0]} rows remain")

# ----- 1.2b  Fix column names -----

df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_", regex=False))
print("\n[COLUMN NAMES CLEANED]:", df.columns.tolist())

target_col = "outcome"    # note the lowercase now

# ----- 1.2c  Correct data types -----

print("\n[DATA TYPES AFTER CLEANING]:")
print(df.dtypes)


# ----- 1.2d  Handle missing values -----

zero_invalid = ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]
print("\n[ZERO VALUE COUNTS] (zeros = invalid in medical context):")
for col in zero_invalid:
    n = (df[col] == 0).sum()
    pct = n / len(df) * 100
    print(f"  {col:25s}: {n:3d} zeros ({pct:.1f}%)")

# Replace 0 → NaN so pandas treats them as missing
df[zero_invalid] = df[zero_invalid].replace(0, np.nan)

# Impute with MEDIAN per column
for col in zero_invalid:
    med = df[col].median()
    df[col] = df[col].fillna(med)
    print(f"  Imputed '{col}' with median = {med:.2f}")

print(f"\n[MISSING VALUES AFTER IMPUTATION]:\n{df.isnull().sum()}")

# ----- 1.2e  Standardise / validate remaining values -----

print("\n[VALUE VALIDATION] — checking plausible ranges...")
print(df[["glucose","bloodpressure","bmi"]].describe().round(2))

# ----- 1.2f  Outlier detection using IQR -----

print("\n[IQR OUTLIER DETECTION]:")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != target_col]

outlier_flags = pd.DataFrame(index=df.index)
total_outliers_per_col = {}

for col in num_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR,  Q3 + 1.5*IQR
    flag = (df[col] < lower) | (df[col] > upper)
    outlier_flags[col + "_outlier"] = flag
    total_outliers_per_col[col] = flag.sum()
    print(f"  {col:30s}: {flag.sum():3d} outliers  "
          f"[{lower:.1f} – {upper:.1f}]")

df = pd.concat([df, outlier_flags], axis=1)
print(f"\n  Total rows with ≥1 outlier flag: "
      f"{outlier_flags.any(axis=1).sum()}")


# ──────────────────────────────────────────────────────────────────────
# 1.3  EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────────────────


print("\n\n── 1.3  EXPLORATORY DATA ANALYSIS (EDA) ──")

# Work on the clean numeric features (no outlier-flag columns)
features = num_cols     # 8 feature columns

# ── A) UNIVARIATE – Distribution plots ──────────────────────────────

print("\n[UNIVARIATE] Distribution plots …")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

skew_vals = {}
kurt_vals = {}

for i, col in enumerate(features):
    ax = axes[i]
    data = df[col]
    sk   = skew(data.dropna())
    ku   = kurtosis(data.dropna())
    skew_vals[col] = sk
    kurt_vals[col] = ku

    sns.histplot(data, kde=True, ax=ax, color="#4C72B0", bins=30)
    ax.set_title(f"{col}\nskew={sk:.2f}  kurt={ku:.2f}", fontsize=10)
    ax.set_xlabel("")

# Last subplot: summary text
axes[-1].axis("off")
axes[-1].text(0.1, 0.5,
    "\n".join([f"{c}: skew={v:.2f}" for c, v in skew_vals.items()]),
    transform=axes[-1].transAxes, fontsize=9, family="monospace",
    verticalalignment="center")
axes[-1].set_title("Skewness Summary", fontsize=10)

plt.suptitle("Univariate Distributions", fontsize=14, y=1.01)
plt.tight_layout()
savefig("univariate_distributions")

# ----- Skewness & Kurtosis table -----

print("\n[SKEWNESS & KURTOSIS]:")
sk_df = pd.DataFrame({"skewness": skew_vals, "kurtosis": kurt_vals})
print(sk_df.round(3).to_string())

# ----- Log transformation for skewed columns -----


skewed_cols = [c for c, v in skew_vals.items() if abs(v) > 1]
print(f"\n[LOG TRANSFORM] Columns with |skew|>1: {skewed_cols}")

df_log = df[features].copy()
for col in skewed_cols:
    df_log[col + "_log"] = np.log1p(df_log[col])

# Plot before/after for insulin (most skewed)
if "insulin" in skewed_cols:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["insulin"], kde=True, ax=ax1, color="#4C72B0", bins=30)
    ax1.set_title("Insulin (original)  skew={:.2f}".format(skew_vals["insulin"]))
    sns.histplot(df_log["insulin_log"], kde=True, ax=ax2, color="#55A868", bins=30)
    ax2.set_title("log(Insulin+1)  skew={:.2f}".format(
        skew(df_log["insulin_log"])))
    plt.suptitle("Log Transformation Effect", fontsize=13)
    plt.tight_layout()
    savefig("log_transformation_insulin")


# ── B) BIVARIATE – Correlation heatmap ──────────────────────────────
print("\n[BIVARIATE] Correlation heatmap …")

corr = df[features + [target_col]].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))   # only lower triangle
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
savefig("correlation_heatmap")

print("  Strong correlations with 'outcome':")
for f in features:
    r = corr.loc[f, target_col]
    if abs(r) > 0.2:
        print(f"    {f:30s}: r={r:.3f}")

# ── C) BIVARIATE – Scatter plots (top 3 features vs target) ─────────

print("\n[BIVARIATE] Scatter plots …")
top3 = corr[target_col].drop(target_col).abs().nlargest(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, top3):
    scatter = ax.scatter(df[col], df[target_col],
                         c=df[target_col], cmap="coolwarm",
                         alpha=0.5, edgecolors="none")
    ax.set_xlabel(col); ax.set_ylabel("outcome")
    ax.set_title(f"{col} vs outcome")
plt.suptitle("Top-3 Feature Scatter Plots vs Target", fontsize=13)
plt.tight_layout()
savefig("scatter_top3_features")

# ── D) BIVARIATE – Boxplots (feature vs target) ─────────────────────
print("\n[BIVARIATE] Boxplots …")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(x=target_col, y=col, data=df,
                palette=["#4C72B0", "#DD8452"],
                ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xticklabels(["No DM", "DM"])
plt.suptitle("Feature Distributions by Diabetes Status", fontsize=13)
plt.tight_layout()
savefig("boxplots_by_class")

# ── E) BIVARIATE – Mean feature value per class (bar) ───────────────
print("\n[BIVARIATE] Mean values per class …")
class_means = df.groupby(target_col)[features].mean()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(features):
    class_means[col].plot(kind="bar", ax=axes[i],
                          color=["#4C72B0", "#DD8452"], edgecolor="k")
    axes[i].set_title(col)
    axes[i].set_xticklabels(["No DM", "DM"], rotation=0)
    axes[i].set_ylabel("Mean")
plt.suptitle("Mean Feature Value by Diabetes Status", fontsize=13)
plt.tight_layout()
savefig("barplots_mean_by_class")


# ── F) MULTIVARIATE – 3D Scatter ────────────────────────────────────

print("\n[MULTIVARIATE] 3D scatter plot …")
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d")
colors = df[target_col].map({0: "#4C72B0", 1: "#DD8452"})
ax.scatter(df["glucose"], df["bmi"], df["age"],
           c=colors, alpha=0.5, s=20)
ax.set_xlabel("Glucose"); ax.set_ylabel("BMI"); ax.set_zlabel("Age")
ax.set_title("3D Scatter: Glucose × BMI × Age\n(blue=no DM, orange=DM)",
             fontsize=12)
plt.tight_layout()
savefig("3d_scatter")

# ── G) MULTIVARIATE – VIF (Multicollinearity) ───────────────────────

print("\n[VIF – MULTICOLLINEARITY CHECK]:")
X_vif = df[features].copy()
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                   for i in range(X_vif.shape[1])]
print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

high_vif = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
print(f"\n  Features with VIF>10 (multicollinearity problem): "
      f"{high_vif if high_vif else 'None — we are good!'}")

# ──────────────────────────────────────────────────────────────────────
# 1.4  DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────


print("\n\n── 1.4  DATA PREPROCESSING ──")

# ----- 1.4a  Encoding categorical variables -----

print("\n[ENCODING] No categorical features in this dataset.")


# ----- 1.4b  Outlier treatment -----

print("\n[OUTLIER TREATMENT] Winsorisation (IQR clipping) …")
df_clean = df[features + [target_col]].copy()   # strip outlier-flag cols

for col in features:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR,  Q3 + 1.5*IQR
    before = df_clean[col].between(lower, upper).sum()
    df_clean[col] = df_clean[col].clip(lower, upper)
    print(f"  {col:30s}: clipped to [{lower:.1f}, {upper:.1f}]")

# ----- 1.4c  Feature scaling -----

print("\n[SCALING] StandardScaler applied …")

X = df_clean[features]
y = df_clean[target_col]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

print("  After scaling — first 3 rows:")
print(X_scaled.head(3).round(3).to_string())

# ----- 1.4d  Handle imbalanced data -----

print("\n[IMBALANCED DATA] Class distribution:")
print(y.value_counts().to_string())

# ----- 1.4e  Train/Test split -----

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n[TRAIN/TEST SPLIT]  train={len(X_train)}  test={len(X_test)}")
print(f"  Train class dist: {y_train.value_counts().to_dict()}")
print(f"  Test  class dist: {y_test.value_counts().to_dict()}")

# ----- Apply oversampling to TRAINING data only -----
# Combine for easy manipulation
X_tr_df = pd.DataFrame(X_train, columns=features)
X_tr_df["outcome"] = y_train.values

minority = X_tr_df[X_tr_df["outcome"] == 1]
majority = X_tr_df[X_tr_df["outcome"] == 0]

minority_up = resample(minority,
                        replace=True,
                        n_samples=len(majority),
                        random_state=42)
X_tr_balanced = pd.concat([majority, minority_up])
X_tr_balanced = X_tr_balanced.sample(frac=1, random_state=42)  # shuffle

X_train_bal = X_tr_balanced[features]
y_train_bal  = X_tr_balanced["outcome"]

print(f"\n  After oversampling — train size: {len(X_train_bal)}")
print(f"  Class dist: {y_train_bal.value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════
#  PART 2 – AI TASKS
# ══════════════════════════════════════════════════════════════════════

print("\n\n" + "="*70)
print("  PART 2 – AI TASKS")
print("="*70)


# ──────────────────────────────────────────────────────────────────────
# 2.1  CLUSTERING (Unsupervised Learning)
# ──────────────────────────────────────────────────────────────────────


print("\n── 2.1  CLUSTERING (KMeans) ──")

# Remove target column for clustering
X_clust = X_scaled.copy()

# ----- Elbow Method -----
print("\n[ELBOW METHOD] Testing k=1 to 10 …")
inertias = []
k_range  = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, inertias, "bo-", markersize=8, linewidth=2)
ax.set_xlabel("Number of Clusters (k)", fontsize=12)
ax.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=12)
ax.set_title("Elbow Method – Choosing Optimal k", fontsize=13)
ax.axvline(x=2, color="red", linestyle="--", label="Elbow at k=2")
ax.legend()
plt.tight_layout()
savefig("kmeans_elbow")

# ----- Fit K=2 -----

print("\n[KMEANS] Fitting with k=2 …")
km_final = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_clust)

# ----- Evaluation -----

sil = silhouette_score(X_clust, cluster_labels)
dbi = davies_bouldin_score(X_clust, cluster_labels)

print(f"  Silhouette Score       : {sil:.4f}  (higher is better, max=1)")
print(f"  Davies-Bouldin Index   : {dbi:.4f}  (lower is better, min=0)")

# ----- Cluster vs true label agreement -----
cluster_outcome = pd.crosstab(cluster_labels, y,
                               rownames=["Cluster"],
                               colnames=["Outcome"])
print("\n[CLUSTER vs TRUE LABEL]:")
print(cluster_outcome)

# ----- PCA 2D cluster plot -----
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_clust)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# KMeans clusters
sc1 = ax1.scatter(X_pca2[:, 0], X_pca2[:, 1],
                  c=cluster_labels, cmap="Set1", alpha=0.6, s=20)
ax1.set_title("KMeans Clusters (k=2)", fontsize=12)
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
plt.colorbar(sc1, ax=ax1, label="Cluster")

# True labels
sc2 = ax2.scatter(X_pca2[:, 0], X_pca2[:, 1],
                  c=y.values, cmap="coolwarm", alpha=0.6, s=20)
ax2.set_title("True Labels (0=no DM, 1=DM)", fontsize=12)
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
plt.colorbar(sc2, ax=ax2, label="Outcome")

plt.suptitle("KMeans Clusters vs True Labels (PCA 2D view)", fontsize=13)
plt.tight_layout()
savefig("kmeans_clusters_vs_truth")


# ──────────────────────────────────────────────────────────────────────
# 2.2  CLASSIFICATION (Supervised Learning)
# ──────────────────────────────────────────────────────────────────────


print("\n── 2.2  CLASSIFICATION ──")

# ----- Define all models to compare -----
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
   
}

results = {}

for name, model in models.items():
    # Train on balanced training data
    model.fit(X_train_bal, y_train_bal)

    # Predict on ORIGINAL (unbalanced) test data
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {"Accuracy": acc, "F1": f1, "ROC-AUC": auc,
                     "y_pred": y_pred, "y_proba": y_proba}
    print(f"\n  [{name}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")

# ----- Detailed report for best model -----
best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
best = results[best_name]

print(f"\n\n{'─'*60}")
print(f"  BEST MODEL: {best_name}  (highest ROC-AUC)")
print(f"{'─'*60}")
print("\n[CLASSIFICATION REPORT]:")
print(classification_report(y_test, best["y_pred"],
                             target_names=["No Diabetes", "Diabetes"]))

# ----- Confusion Matrix -----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No DM", "DM"])
    disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
    axes[i].set_title(f"{name}\nAcc={res['Accuracy']:.3f}  "
                      f"F1={res['F1']:.3f}  AUC={res['ROC-AUC']:.3f}",
                      fontsize=10)

axes[-1].axis("off")
plt.suptitle("Confusion Matrices – All Models", fontsize=14)
plt.tight_layout()
savefig("confusion_matrices_all_models")

# ----- ROC Curves for all models -----
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{name}  (AUC={res['ROC-AUC']:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves – All Models", fontsize=13)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
savefig("roc_curves_all_models")

# ----- Model Comparison Bar Chart -----
results_df = pd.DataFrame({k: {m: v for m, v in v.items()
                                if m in ["Accuracy","F1","ROC-AUC"]}
                            for k, v in results.items()}).T

print("\n[MODEL COMPARISON TABLE]:")
print(results_df.round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 6))
results_df[["Accuracy","F1","ROC-AUC"]].plot(
    kind="bar", ax=ax,
    color=["#4C72B0","#DD8452","#55A868"],
    edgecolor="k", width=0.7)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison: Accuracy / F1 / ROC-AUC", fontsize=13)
ax.set_xticklabels(results_df.index, rotation=15, ha="right")
ax.legend(loc="lower right")
ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="0.8 threshold")
plt.tight_layout()
savefig("model_comparison_bar")

# ----- Feature Importance (Random Forest) -----
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=features)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="k")
ax.set_title("Random Forest – Feature Importances", fontsize=13)
ax.set_xlabel("Importance Score")
plt.tight_layout()
savefig("random_forest_feature_importance")

print("\n[FEATURE IMPORTANCE – Random Forest]:")
print(importances.sort_values(ascending=False).round(4).to_string())


# ══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("  FINAL SUMMARY")
print("="*70)

print(f"""
  DATASET  : Pima Indians Diabetes (768 rows, 8 features)
  TARGET   : Outcome (0=no diabetes, 1=diabetes) — 65:35 imbalance

  KEY FINDINGS (EDA):
    • Glucose, BMI, Age are the top 3 predictors of diabetes
    • Insulin and SkinThickness had ~50% missing/invalid zeros → median-imputed
    • Dataset is moderately imbalanced → addressed with oversampling

  CLUSTERING:
    • KMeans k=2 partially recovers the diabetic/non-diabetic groups
    • Silhouette Score = {sil:.3f} — moderate cluster quality
    • (Diabetes doesn't cluster perfectly because it's not purely
      linear — there's natural overlap between groups)

  CLASSIFICATION RESULTS:
""")

for name in results:
    r = results[name]
    marker = " ← BEST" if name == best_name else ""
    print(f"    {name:30s}  "
          f"Acc={r['Accuracy']:.3f}  F1={r['F1']:.3f}  "
          f"AUC={r['ROC-AUC']:.3f}{marker}")

print(f"""
  WHY {best_name.upper()} IS BEST:
    • Highest ROC-AUC → best at separating diabetic from non-diabetic
      across all decision thresholds
    • Ensemble of many decision trees → robust to noise and outliers
    • Handles non-linear relationships naturally
    • Less sensitive to feature scaling than Logistic Regression
    • No assumptions about data distribution

 
""")

joblib.dump(models[best_name], "./outputs/best_model.pkl")

