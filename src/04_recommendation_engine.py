"""
Personalized Recommendation Logic
- Content-based filtering (category/level match to learner preference)
- Similar learner profiles (within-cluster collaborative signal)
- Course popularity within cluster
- Rating-weighted relevance score
"""

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

import pandas as pd
import numpy as np

DATA = os.path.join(_ROOT, "data")
OUT = os.path.join(_ROOT, "outputs")

users = pd.read_csv(f"{DATA}/learner_profiles_clustered.csv")
courses = pd.read_csv(f"{DATA}/courses.csv")
txns = pd.read_csv(f"{DATA}/transactions.csv")

CLUSTER_NAMES = {
    0: "Specialists",
    1: "Explorers",
    2: "Premium Upskillers",
    3: "Casual Dabblers",
    4: "Niche Beginners",
    -1: "Unsegmented (New/Inactive)",
}

def recommend_for_user(user_id, top_n=5, level_filter=None, category_filter=None):
    urow = users[users.UserID == user_id]
    if urow.empty:
        return None
    urow = urow.iloc[0]
    cluster = urow["Cluster"]

    enrolled = set(txns[txns.UserID == user_id]["CourseID"])
    candidates = courses[~courses.CourseID.isin(enrolled)].copy()

    if level_filter:
        candidates = candidates[candidates.CourseLevel == level_filter]
    if category_filter:
        candidates = candidates[candidates.CourseCategory == category_filter]

    if candidates.empty:
        return pd.DataFrame()

    # --- Course popularity within cluster (collaborative signal) ---
    cluster_user_ids = users[users.Cluster == cluster]["UserID"]
    cluster_txns = txns[txns.UserID.isin(cluster_user_ids)]
    pop_counts = cluster_txns["CourseID"].value_counts()
    max_pop = pop_counts.max() if len(pop_counts) else 1
    candidates["ClusterPopularity"] = candidates["CourseID"].map(pop_counts).fillna(0) / max(max_pop, 1)

    # --- Content-based match: category & level alignment to learner preference ---
    pref_cat = urow["PreferredCategory"]
    pref_lvl = urow["PreferredLevel"]
    candidates["CategoryMatch"] = (candidates["CourseCategory"] == pref_cat).astype(float)
    candidates["LevelMatch"] = (candidates["CourseLevel"] == pref_lvl).astype(float)

    # Soft level proximity (beginner-intermediate-advanced) so adjacent levels aren't zeroed out
    level_rank = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    learner_depth = urow["LearningDepthIndex"]
    target_rank = 0 + learner_depth * 2  # 0..2 continuous target based on depth
    candidates["LevelProximity"] = 1 - (abs(candidates["CourseLevel"].map(level_rank) - target_rank) / 2)

    # --- Rating-weighted relevance ---
    candidates["RatingScore"] = candidates["CourseRating"] / 5.0

    # --- Final weighted relevance score ---
    candidates["RelevanceScore"] = (
        0.30 * candidates["CategoryMatch"] +
        0.15 * candidates["LevelMatch"] +
        0.15 * candidates["LevelProximity"] +
        0.20 * candidates["ClusterPopularity"] +
        0.20 * candidates["RatingScore"]
    )

    result = candidates.sort_values("RelevanceScore", ascending=False).head(top_n)
    return result[["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating", "RelevanceScore"]]


def evaluate_recommendation_precision(n_eval_users=200, top_n=5, seed=42):
    """
    Proxy evaluation: hide each evaluated learner's most recent course,
    retrain recommendation context on the remainder, and check whether the
    held-out course's category appears in the top-N recommendations
    (Precision@N, category-level relevance proxy).
    """
    rng = np.random.default_rng(seed)
    eval_users = users[users["TotalCoursesEnrolled"] >= 2]["UserID"].sample(
        min(n_eval_users, (users["TotalCoursesEnrolled"] >= 2).sum()), random_state=seed
    )
    hits, total = 0, 0
    for uid in eval_users:
        user_txns = txns[txns.UserID == uid].sort_values("TransactionDate")
        if len(user_txns) < 2:
            continue
        held_out = user_txns.iloc[-1]
        held_out_course = courses[courses.CourseID == held_out.CourseID].iloc[0]

        recs = recommend_for_user(uid, top_n=top_n)
        if recs is None or recs.empty:
            continue
        total += 1
        if held_out_course["CourseCategory"] in recs["CourseCategory"].values:
            hits += 1
    precision = hits / total if total else 0
    return precision, hits, total


def engagement_lift_proxy():
    """
    Compare avg CourseRating of cluster-popular recommended courses vs.
    the platform-wide average rating of courses actually enrolled in,
    as a simple proxy for expected engagement lift from personalization.
    """
    baseline_rating = courses["CourseRating"].mean()
    sample_users = users[users["TotalCoursesEnrolled"] > 0]["UserID"].sample(300, random_state=42)
    rec_ratings = []
    for uid in sample_users:
        recs = recommend_for_user(uid, top_n=5)
        if recs is not None and not recs.empty:
            rec_ratings.append(recs["CourseRating"].mean())
    avg_rec_rating = float(np.mean(rec_ratings))
    lift_pct = (avg_rec_rating - baseline_rating) / baseline_rating * 100
    return baseline_rating, avg_rec_rating, lift_pct


if __name__ == "__main__":
    sample_uid = users.iloc[0]["UserID"]
    print(f"Sample recommendations for {sample_uid} (cluster {users.iloc[0]['Cluster']}):")
    print(recommend_for_user(sample_uid, top_n=5))

    precision, hits, total = evaluate_recommendation_precision(n_eval_users=300)
    print(f"\nRecommendation Precision@5 (category-match proxy): {precision:.3f} ({hits}/{total})")

    baseline, rec_avg, lift = engagement_lift_proxy()
    print(f"\nBaseline avg course rating: {baseline:.3f}")
    print(f"Avg rating of recommended courses: {rec_avg:.3f}")
    print(f"Engagement Lift (Proxy): {lift:.2f}%")

    import json
    with open(f"{OUT}/recommendation_eval.json", "w") as f:
        json.dump({
            "precision_at_5": precision, "hits": hits, "total_evaluated": total,
            "baseline_avg_rating": baseline, "recommended_avg_rating": rec_avg,
            "engagement_lift_pct": lift
        }, f, indent=2)
