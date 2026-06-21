"""
Evaluation & Validation
- Silhouette Score (already computed in 03)
- Intra-Cluster Similarity (behavioral consistency)
- Recommendation Precision (already computed in 04)
- Engagement Lift Proxy (already computed in 04)
"""

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json

DATA = os.path.join(_ROOT, "data")
OUT = os.path.join(_ROOT, "outputs")

df = pd.read_csv(f"{DATA}/learner_profiles_clustered.csv")
df = df[df["Cluster"] != -1].copy()

numeric_features = [
    "Age", "TotalCoursesEnrolled", "NumCategoriesExplored", "AvgCoursesPerCategory",
    "EnrollmentFrequency", "AvgCourseRatingEnrolled", "AvgSpendPerLearner",
    "DiversityScore", "LearningDepthIndex", "BeginnerShare", "AdvancedShare",
    "CertificationShare"
]
X = StandardScaler().fit_transform(df[numeric_features])
df_idx = df.reset_index(drop=True)

intra_sims = {}
rng = np.random.default_rng(42)
for c in sorted(df_idx["Cluster"].unique()):
    idx = df_idx[df_idx["Cluster"] == c].index.values
    if len(idx) > 400:  # sample for tractability
        idx = rng.choice(idx, 400, replace=False)
    sub_X = X[idx]
    sim_matrix = cosine_similarity(sub_X)
    # average of upper triangle excluding diagonal
    n = sim_matrix.shape[0]
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    intra_sims[int(c)] = float(avg_sim)

# Inter-cluster (control) similarity for comparison: random pairs across different clusters
cross_sims = []
clusters_list = sorted(df_idx["Cluster"].unique())
for i in range(2000):
    c1, c2 = rng.choice(clusters_list, 2, replace=False)
    u1 = df_idx[df_idx["Cluster"] == c1].sample(1, random_state=int(rng.integers(0, 1e6))).index[0]
    u2 = df_idx[df_idx["Cluster"] == c2].sample(1, random_state=int(rng.integers(0, 1e6))).index[0]
    cross_sims.append(cosine_similarity(X[[u1]], X[[u2]])[0, 0])
avg_cross_sim = float(np.mean(cross_sims))

print("Intra-cluster similarity (avg cosine):")
for c, s in intra_sims.items():
    print(f"  Cluster {c}: {s:.3f}")
print(f"Average cross-cluster similarity (control): {avg_cross_sim:.3f}")

with open(f"{OUT}/clustering_meta.json") as f:
    meta = json.load(f)
with open(f"{OUT}/recommendation_eval.json") as f:
    rec_eval = json.load(f)

evaluation_summary = {
    "silhouette_score_final": meta["final_silhouette"],
    "hierarchical_silhouette": meta["hier_silhouette"],
    "ari_kmeans_vs_hierarchical": meta["ari_kmeans_vs_hier"],
    "intra_cluster_similarity": intra_sims,
    "avg_cross_cluster_similarity": avg_cross_sim,
    "recommendation_precision_at_5": rec_eval["precision_at_5"],
    "engagement_lift_pct_proxy": rec_eval["engagement_lift_pct"],
}

with open(f"{OUT}/evaluation_summary.json", "w") as f:
    json.dump(evaluation_summary, f, indent=2)

print("\nFull evaluation summary saved to evaluation_summary.json")
print(json.dumps(evaluation_summary, indent=2))
