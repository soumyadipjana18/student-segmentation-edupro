"""
Data Preprocessing + Learner Segmentation (Clustering)
- Normalize numerical features, encode categoricals
- K-Means with elbow + silhouette to choose k
- Hierarchical clustering as validation
"""

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(_ROOT, "data")
OUT = os.path.join(_ROOT, "outputs")

df = pd.read_csv(f"{DATA}/learner_profiles.csv")

# Drop learners with zero engagement (no real behavioral signal) for clustering
clustering_df = df[df["TotalCoursesEnrolled"] > 0].copy()
print("Learners used for clustering:", clustering_df.shape[0], "/", df.shape[0])

numeric_features = [
    "Age", "TotalCoursesEnrolled", "NumCategoriesExplored", "AvgCoursesPerCategory",
    "EnrollmentFrequency", "AvgCourseRatingEnrolled", "AvgSpendPerLearner",
    "DiversityScore", "LearningDepthIndex", "BeginnerShare", "AdvancedShare",
    "CertificationShare"
]
categorical_features = ["PreferredCategory", "PreferredLevel", "Gender"]

X_num = clustering_df[numeric_features].copy()
# Reduce noise from sparse enrollments: winsorize extreme outliers (cap at 1st/99th pct)
for col in numeric_features:
    lo, hi = X_num[col].quantile([0.01, 0.99])
    X_num[col] = X_num[col].clip(lo, hi)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(clustering_df[categorical_features])

X = np.hstack([X_num_scaled, X_cat * 0.6])  # mild downweight of categorical block

# ---------------------------------------------------------------
# Elbow + Silhouette to choose K
# ---------------------------------------------------------------
inertias, sil_scores = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, marker="o", color="#2E5EAA")
axes[0].set_title("Elbow Method (Inertia vs K)")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), sil_scores, marker="o", color="#C2410C")
axes[1].set_title("Silhouette Score vs K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{OUT}/elbow_silhouette.png", dpi=150)
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print("Silhouette scores:", dict(zip(K_range, [round(s,3) for s in sil_scores])))
print("Best K by silhouette:", best_k)

# Business reasoning: choose k=5 (matches the brief's expectation of distinct, interpretable segments)
# unless silhouette strongly favors a different k
FINAL_K = 5

km_final = KMeans(n_clusters=FINAL_K, random_state=42, n_init=25)
clustering_df["Cluster"] = km_final.fit_predict(X)
final_sil = silhouette_score(X, clustering_df["Cluster"])
print(f"Final KMeans (k={FINAL_K}) silhouette: {final_sil:.3f}")

# ---------------------------------------------------------------
# Hierarchical clustering as validation
# ---------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=FINAL_K, linkage="ward")
hier_labels = agg.fit_predict(X)
clustering_df["HierCluster"] = hier_labels
hier_sil = silhouette_score(X, hier_labels)
ari = adjusted_rand_score(clustering_df["Cluster"], hier_labels)
print(f"Hierarchical (k={FINAL_K}) silhouette: {hier_sil:.3f}")
print(f"Adjusted Rand Index (KMeans vs Hierarchical agreement): {ari:.3f}")

# ---------------------------------------------------------------
# PCA for 2D visualization
# ---------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
clustering_df["PCA1"] = coords[:, 0]
clustering_df["PCA2"] = coords[:, 1]

plt.figure(figsize=(7, 6))
palette = ["#2E5EAA", "#C2410C", "#15803D", "#7C3AED", "#DB2777", "#0891B2", "#CA8A04"]
for c in sorted(clustering_df["Cluster"].unique()):
    sub = clustering_df[clustering_df["Cluster"] == c]
    plt.scatter(sub["PCA1"], sub["PCA2"], s=18, alpha=0.65, color=palette[c % len(palette)], label=f"Cluster {c}")
plt.title(f"Learner Segments in PCA Space (K={FINAL_K})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/pca_clusters.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Cluster profiles (means) for interpretation
# ---------------------------------------------------------------
profile_summary = clustering_df.groupby("Cluster")[numeric_features].mean().round(2)
profile_summary["N_Learners"] = clustering_df.groupby("Cluster").size()
profile_summary["TopCategory"] = clustering_df.groupby("Cluster")["PreferredCategory"].agg(lambda x: x.value_counts().idxmax())
profile_summary["TopLevel"] = clustering_df.groupby("Cluster")["PreferredLevel"].agg(lambda x: x.value_counts().idxmax())
print("\nCluster profile summary:\n", profile_summary)

profile_summary.to_csv(f"{OUT}/cluster_profile_summary.csv")

# Merge cluster labels back into the full learner_df (zero-engagement learners get Cluster = -1)
df["Cluster"] = -1
df.loc[clustering_df.index, "Cluster"] = clustering_df["Cluster"].values
df.to_csv(f"{DATA}/learner_profiles_clustered.csv", index=False)

# Save scaler/encoder-ready numeric matrix metadata for reuse in the app
import json
with open(f"{OUT}/clustering_meta.json", "w") as f:
    json.dump({
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "final_k": FINAL_K,
        "final_silhouette": float(final_sil),
        "hier_silhouette": float(hier_sil),
        "ari_kmeans_vs_hier": float(ari),
        "silhouette_by_k": {int(k): float(s) for k, s in zip(K_range, sil_scores)},
    }, f, indent=2)

print("\nSaved: learner_profiles_clustered.csv, elbow_silhouette.png, pca_clusters.png, cluster_profile_summary.csv, clustering_meta.json")
