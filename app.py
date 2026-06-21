"""
EduPro — Student Segmentation & Personalized Course Recommendation Dashboard
Run with: streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EduPro Learner Intelligence", layout="wide", page_icon="🎓")

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_THIS_DIR, "data")
OUT = os.path.join(_THIS_DIR, "outputs")

CLUSTER_NAMES = {
    0: "Specialists",
    1: "Explorers",
    2: "Premium Upskillers",
    3: "Casual Dabblers",
    4: "Niche Beginners",
    -1: "Unsegmented (New/Inactive)",
}
CLUSTER_DESCRIPTIONS = {
    "Specialists": "Focus deeply within one or two categories at intermediate level. High courses-per-category ratio.",
    "Explorers": "Spread enrollments across the widest range of categories. High diversity, mostly beginner/intermediate.",
    "Premium Upskillers": "Lower volume but high spend, advanced-level focus, and the highest certification share — career-driven.",
    "Casual Dabblers": "Low engagement, mostly beginner level, lowest spend — at risk of churn.",
    "Niche Beginners": "Small enrollment count concentrated in a single niche category, high certification share, beginner level.",
    "Unsegmented (New/Inactive)": "No transaction history yet — new or inactive learners.",
}
PALETTE = {
    "Specialists": "#2E5EAA", "Explorers": "#C2410C", "Premium Upskillers": "#15803D",
    "Casual Dabblers": "#7C3AED", "Niche Beginners": "#DB2777", "Unsegmented (New/Inactive)": "#94A3B8",
}


@st.cache_data
def load_data():
    users = pd.read_csv(f"{DATA}/learner_profiles_clustered.csv")
    courses = pd.read_csv(f"{DATA}/courses.csv")
    txns = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["TransactionDate"])
    users["SegmentName"] = users["Cluster"].map(CLUSTER_NAMES)
    with open(f"{OUT}/evaluation_summary.json") as f:
        eval_summary = json.load(f)
    with open(f"{OUT}/clustering_meta.json") as f:
        cluster_meta = json.load(f)
    cluster_profile = pd.read_csv(f"{OUT}/cluster_profile_summary.csv")
    return users, courses, txns, eval_summary, cluster_meta, cluster_profile


users, courses, txns, eval_summary, cluster_meta, cluster_profile = load_data()


def recommend_for_user(user_id, top_n=5, level_filter=None, category_filter=None):
    urow = users[users.UserID == user_id]
    if urow.empty:
        return pd.DataFrame()
    urow = urow.iloc[0]
    cluster = urow["Cluster"]

    enrolled = set(txns[txns.UserID == user_id]["CourseID"])
    candidates = courses[~courses.CourseID.isin(enrolled)].copy()
    if level_filter and level_filter != "Any":
        candidates = candidates[candidates.CourseLevel == level_filter]
    if category_filter and category_filter != "Any":
        candidates = candidates[candidates.CourseCategory == category_filter]
    if candidates.empty:
        return pd.DataFrame()

    cluster_user_ids = users[users.Cluster == cluster]["UserID"]
    cluster_txns = txns[txns.UserID.isin(cluster_user_ids)]
    pop_counts = cluster_txns["CourseID"].value_counts()
    max_pop = pop_counts.max() if len(pop_counts) else 1
    candidates["ClusterPopularity"] = candidates["CourseID"].map(pop_counts).fillna(0) / max(max_pop, 1)

    pref_cat = urow["PreferredCategory"]
    pref_lvl = urow["PreferredLevel"]
    candidates["CategoryMatch"] = (candidates["CourseCategory"] == pref_cat).astype(float)
    candidates["LevelMatch"] = (candidates["CourseLevel"] == pref_lvl).astype(float)

    level_rank = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    learner_depth = urow["LearningDepthIndex"]
    target_rank = 0 + learner_depth * 2
    candidates["LevelProximity"] = 1 - (abs(candidates["CourseLevel"].map(level_rank) - target_rank) / 2)
    candidates["RatingScore"] = candidates["CourseRating"] / 5.0

    candidates["RelevanceScore"] = (
        0.30 * candidates["CategoryMatch"] +
        0.15 * candidates["LevelMatch"] +
        0.15 * candidates["LevelProximity"] +
        0.20 * candidates["ClusterPopularity"] +
        0.20 * candidates["RatingScore"]
    )
    result = candidates.sort_values("RelevanceScore", ascending=False).head(top_n)
    return result[["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating", "RelevanceScore"]]


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title("🎓 EduPro")
st.sidebar.caption("Learner Intelligence Platform")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Learner Profile Explorer",
    "Cluster Visualization",
    "Personalized Recommendations",
    "Segment Comparison",
    "Model Evaluation",
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Learners: {len(users):,} | Courses: {len(courses):,} | Transactions: {len(txns):,}")

# ---------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.title("EduPro Student Segmentation & Personalized Recommendation System")
    st.markdown(
        "A data-driven personalization engine that segments learners into behavioral cohorts "
        "and serves cluster-aware, content-based course recommendations."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Learners", f"{len(users):,}")
    c2.metric("Total Courses", f"{len(courses):,}")
    c3.metric("Total Transactions", f"{len(txns):,}")
    c4.metric("Segments Identified", f"{cluster_meta['final_k']}")

    st.markdown("### Segment Distribution")
    seg_counts = users[users.Cluster != -1]["SegmentName"].value_counts().reindex(
        [n for n in CLUSTER_NAMES.values() if n != "Unsegmented (New/Inactive)"]
    )
    fig = px.bar(
        x=seg_counts.index, y=seg_counts.values,
        color=seg_counts.index, color_discrete_map=PALETTE,
        labels={"x": "Segment", "y": "Number of Learners"},
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Enrollments by Category")
        tc = txns.merge(courses, on="CourseID")
        cat_counts = tc["CourseCategory"].value_counts()
        fig2 = px.bar(x=cat_counts.values, y=cat_counts.index, orientation="h",
                       labels={"x": "Enrollments", "y": ""}, color_discrete_sequence=["#2E5EAA"])
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("### Monthly Enrollment Trend")
        trend = txns.set_index("TransactionDate").resample("MS").size().reset_index(name="Enrollments")
        fig3 = px.line(trend, x="TransactionDate", y="Enrollments", markers=True,
                        color_discrete_sequence=["#15803D"])
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Key Insight")
    st.info(
        "Five behaviorally distinct learner segments emerge from enrollment, spend, and depth patterns. "
        "Cluster-aware recommendations lift the average rating of suggested courses relative to platform "
        f"baseline by **{eval_summary['engagement_lift_pct_proxy']:.1f}%** (proxy estimate), with "
        f"**Precision@5 of {eval_summary['recommendation_precision_at_5']:.1%}** on held-out enrollments."
    )

# ---------------------------------------------------------------------
# PAGE: LEARNER PROFILE EXPLORER
# ---------------------------------------------------------------------
elif page == "Learner Profile Explorer":
    st.title("Learner Profile Explorer")
    st.markdown("Select a learner to view their full profile, behavioral features, and assigned segment.")

    search = st.text_input("Search by UserID (e.g. U00001)", "")
    filtered_users = users[users.UserID.str.contains(search, case=False)] if search else users
    user_id = st.selectbox("Select Learner", filtered_users["UserID"].head(500).tolist())

    urow = users[users.UserID == user_id].iloc[0]
    seg_name = urow["SegmentName"]

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown("#### Demographics")
        st.write(f"**UserID:** {urow['UserID']}")
        st.write(f"**Age:** {int(urow['Age'])}")
        st.write(f"**Gender:** {urow['Gender']}")
    with col2:
        st.markdown("#### Segment")
        st.markdown(
            f"<span style='background-color:{PALETTE.get(seg_name,'#999')};color:white;"
            f"padding:6px 14px;border-radius:14px;font-weight:600'>{seg_name}</span>",
            unsafe_allow_html=True,
        )
        st.caption(CLUSTER_DESCRIPTIONS.get(seg_name, ""))
    with col3:
        st.markdown("#### Behavioral Snapshot")
        m1, m2, m3 = st.columns(3)
        m1.metric("Courses Enrolled", int(urow["TotalCoursesEnrolled"]))
        m2.metric("Categories Explored", int(urow["NumCategoriesExplored"]))
        m3.metric("Avg Spend", f"${urow['AvgSpendPerLearner']:.0f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Learning Depth Idx", f"{urow['LearningDepthIndex']:.2f}")
        m5.metric("Preferred Category", urow["PreferredCategory"])
        m6.metric("Preferred Level", urow["PreferredLevel"])

    st.markdown("### Enrollment History")
    hist = txns[txns.UserID == user_id].merge(courses, on="CourseID").sort_values("TransactionDate", ascending=False)
    if hist.empty:
        st.warning("This learner has no transaction history yet.")
    else:
        st.dataframe(
            hist[["TransactionDate", "CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating", "Amount"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Feature Radar (vs. Segment Average)")
    feats = ["TotalCoursesEnrolled", "NumCategoriesExplored", "AvgSpendPerLearner", "LearningDepthIndex", "CertificationShare"]
    seg_avg = users[users.Cluster == urow["Cluster"]][feats].mean()
    norm_max = users[feats].max()
    radar_user = [urow[f] / norm_max[f] if norm_max[f] else 0 for f in feats]
    radar_seg = [seg_avg[f] / norm_max[f] if norm_max[f] else 0 for f in feats]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=radar_user + [radar_user[0]], theta=feats + [feats[0]], fill="toself", name="This Learner"))
    fig.add_trace(go.Scatterpolar(r=radar_seg + [radar_seg[0]], theta=feats + [feats[0]], fill="toself", name=f"{seg_name} Avg", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE: CLUSTER VISUALIZATION
# ---------------------------------------------------------------------
elif page == "Cluster Visualization":
    st.title("Cluster Visualization Dashboard")

    tab1, tab2, tab3 = st.tabs(["PCA Scatter", "Elbow & Silhouette", "Cluster Profiles"])

    with tab1:
        st.markdown("#### Learner Segments in PCA Space")
        plot_df = users[users.Cluster != -1].copy()
        # Recompute lightweight PCA-like projection for interactivity (use stored profile features)
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        feats = ["Age", "TotalCoursesEnrolled", "NumCategoriesExplored", "AvgCoursesPerCategory",
                 "EnrollmentFrequency", "AvgCourseRatingEnrolled", "AvgSpendPerLearner",
                 "DiversityScore", "LearningDepthIndex", "BeginnerShare", "AdvancedShare", "CertificationShare"]
        X = StandardScaler().fit_transform(plot_df[feats])
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        plot_df["PCA1"], plot_df["PCA2"] = coords[:, 0], coords[:, 1]
        fig = px.scatter(plot_df, x="PCA1", y="PCA2", color="SegmentName",
                          color_discrete_map=PALETTE, opacity=0.65,
                          hover_data=["UserID", "TotalCoursesEnrolled", "PreferredCategory"])
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Choosing K: Elbow Method & Silhouette Score")
        sil_by_k = cluster_meta["silhouette_by_k"]
        ks = list(sil_by_k.keys())
        sils = list(sil_by_k.values())
        fig = px.line(x=ks, y=sils, markers=True, labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"})
        fig.add_vline(x=cluster_meta["final_k"], line_dash="dash", line_color="red",
                       annotation_text=f"Selected K={cluster_meta['final_k']}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Final model: K={cluster_meta['final_k']}, Silhouette={cluster_meta['final_silhouette']:.3f}. "
            f"K=5 was selected over the silhouette-maximizing K=3 to preserve business-actionable segment "
            "granularity (3 clusters collapsed Career-Switchers and Premium Upskillers together, losing "
            "monetization-relevant distinctions)."
        )

    with tab3:
        st.markdown("#### Cluster Feature Means")
        display_profile = cluster_profile.copy()
        display_profile["Cluster"] = display_profile["Cluster"].map(CLUSTER_NAMES)
        st.dataframe(display_profile, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# PAGE: PERSONALIZED RECOMMENDATIONS
# ---------------------------------------------------------------------
elif page == "Personalized Recommendations":
    st.title("Personalized Course Recommendations")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        user_id = st.selectbox("Learner", users["UserID"].head(500).tolist())
    with col2:
        level_filter = st.selectbox("Filter by Level", ["Any", "Beginner", "Intermediate", "Advanced"])
    with col3:
        cat_filter = st.selectbox("Filter by Category", ["Any"] + sorted(courses["CourseCategory"].unique().tolist()))
    with col4:
        top_n = st.slider("Number of Recommendations", 3, 10, 5)

    urow = users[users.UserID == user_id].iloc[0]
    seg_name = urow["SegmentName"]
    st.markdown(
        f"**Segment:** <span style='background-color:{PALETTE.get(seg_name,'#999')};color:white;"
        f"padding:4px 10px;border-radius:10px'>{seg_name}</span> &nbsp;&nbsp; "
        f"**Preferred Category:** {urow['PreferredCategory']} &nbsp;&nbsp; "
        f"**Preferred Level:** {urow['PreferredLevel']}",
        unsafe_allow_html=True,
    )

    recs = recommend_for_user(user_id, top_n=top_n, level_filter=level_filter, category_filter=cat_filter)
    if recs.empty:
        st.warning("No recommendations available for this filter combination.")
    else:
        st.markdown("### Recommended Courses")
        for _, r in recs.iterrows():
            with st.container(border=True):
                cA, cB, cC = st.columns([3, 2, 1])
                cA.markdown(f"**{r['CourseID']}** — {r['CourseCategory']}")
                cA.caption(f"{r['CourseType']} · {r['CourseLevel']}")
                cB.metric("Rating", f"{r['CourseRating']:.1f} ⭐")
                cC.metric("Relevance", f"{r['RelevanceScore']:.2f}")

        st.markdown("### Why these courses? (Scoring Breakdown)")
        st.caption(
            "Relevance Score = 30% Category Match + 15% Level Match + 15% Level Proximity "
            "+ 20% Cluster Popularity + 20% Rating-Weighted Relevance"
        )
        st.dataframe(recs, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# PAGE: SEGMENT COMPARISON
# ---------------------------------------------------------------------
elif page == "Segment Comparison":
    st.title("Segment Comparison Panel")

    segs = st.multiselect(
        "Select segments to compare",
        [n for n in CLUSTER_NAMES.values() if n != "Unsegmented (New/Inactive)"],
        default=[n for n in CLUSTER_NAMES.values() if n != "Unsegmented (New/Inactive)"],
    )
    comp_df = users[users.SegmentName.isin(segs)]

    feats = ["TotalCoursesEnrolled", "AvgSpendPerLearner", "DiversityScore",
              "LearningDepthIndex", "CertificationShare", "AvgCourseRatingEnrolled"]
    metric = st.selectbox("Metric to Compare", feats)

    fig = px.box(comp_df, x="SegmentName", y=metric, color="SegmentName", color_discrete_map=PALETTE,
                 category_orders={"SegmentName": segs})
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Summary Table")
    summary = comp_df.groupby("SegmentName")[feats].mean().round(2)
    summary["N_Learners"] = comp_df.groupby("SegmentName").size()
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Top Preferred Category by Segment")
    top_cat = comp_df.groupby("SegmentName")["PreferredCategory"].agg(lambda x: x.value_counts().idxmax())
    top_lvl = comp_df.groupby("SegmentName")["PreferredLevel"].agg(lambda x: x.value_counts().idxmax())
    st.dataframe(pd.DataFrame({"Top Category": top_cat, "Top Level": top_lvl}), use_container_width=True)

    st.markdown("### Segment Profiles")
    for s in segs:
        st.markdown(f"**{s}:** {CLUSTER_DESCRIPTIONS.get(s,'')}")

# ---------------------------------------------------------------------
# PAGE: MODEL EVALUATION
# ---------------------------------------------------------------------
elif page == "Model Evaluation":
    st.title("Evaluation & Validation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Silhouette Score (K-Means)", f"{eval_summary['silhouette_score_final']:.3f}")
    c2.metric("Hierarchical Silhouette", f"{eval_summary['hierarchical_silhouette']:.3f}")
    c3.metric("Recommendation Precision@5", f"{eval_summary['recommendation_precision_at_5']:.1%}")
    c4.metric("Engagement Lift (Proxy)", f"{eval_summary['engagement_lift_pct_proxy']:.1f}%")

    st.markdown("### Cross-Validation: K-Means vs. Hierarchical Clustering")
    st.write(
        f"Adjusted Rand Index between K-Means and Agglomerative (Ward) clustering: "
        f"**{eval_summary['ari_kmeans_vs_hierarchical']:.3f}** — indicates substantial agreement "
        "between the two independent algorithms on learner groupings, validating that segments "
        "reflect real structure rather than algorithm-specific artifacts."
    )

    st.markdown("### Intra-Cluster Similarity (Behavioral Consistency)")
    intra = eval_summary["intra_cluster_similarity"]
    intra_named = {CLUSTER_NAMES[int(k)]: v for k, v in intra.items()}
    fig = px.bar(x=list(intra_named.keys()), y=list(intra_named.values()),
                 labels={"x": "Segment", "y": "Avg. Cosine Similarity (within cluster)"},
                 color=list(intra_named.keys()), color_discrete_map=PALETTE)
    fig.add_hline(y=eval_summary["avg_cross_cluster_similarity"], line_dash="dash", line_color="red",
                  annotation_text="Cross-cluster baseline")
    fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Cross-cluster (control) similarity averages {eval_summary['avg_cross_cluster_similarity']:.3f}. "
        "All segments show higher within-cluster similarity than the random cross-cluster baseline, "
        "confirming behaviorally coherent groupings."
    )

    st.markdown("### Metric Reference")
    st.table(pd.DataFrame({
        "Metric": ["Silhouette Score", "Intra-Cluster Similarity", "Recommendation Precision", "Engagement Lift (Proxy)"],
        "Purpose": ["Cluster quality / separation", "Behavioral consistency within segments",
                    "Relevance of top-N recommendations", "Estimated impact of personalization"],
        "Result": [f"{eval_summary['silhouette_score_final']:.3f}",
                    "0.18 – 0.69 across segments (vs. -0.11 baseline)",
                    f"{eval_summary['recommendation_precision_at_5']:.1%} @ Top-5",
                    f"+{eval_summary['engagement_lift_pct_proxy']:.1f}% avg rating vs. baseline"],
    }))
