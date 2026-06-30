"""
EduPro — Student Segmentation & Personalized Course Recommendation Dashboard
Flask version (Vercel-compatible). Run locally with: python app.py
Deploy on Vercel: this file exports a top-level WSGI `app` object.
"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template_string, request

# ---------------------------------------------------------------------
# PATHS / CONSTANTS
# ---------------------------------------------------------------------
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

PAGES = [
    "Overview",
    "Learner Profile Explorer",
    "Cluster Visualization",
    "Personalized Recommendations",
    "Segment Comparison",
    "Model Evaluation",
]

app = Flask(__name__)

# ---------------------------------------------------------------------
# DATA LOADING (cached in memory once per cold start)
# ---------------------------------------------------------------------
_CACHE = {}


def load_data():
    if "data" in _CACHE:
        return _CACHE["data"]
    users = pd.read_csv(f"{DATA}/learner_profiles_clustered.csv")
    courses = pd.read_csv(f"{DATA}/courses.csv")
    txns = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["TransactionDate"])
    users["SegmentName"] = users["Cluster"].map(CLUSTER_NAMES)
    with open(f"{OUT}/evaluation_summary.json") as f:
        eval_summary = json.load(f)
    with open(f"{OUT}/clustering_meta.json") as f:
        cluster_meta = json.load(f)
    cluster_profile = pd.read_csv(f"{OUT}/cluster_profile_summary.csv")
    _CACHE["data"] = (users, courses, txns, eval_summary, cluster_meta, cluster_profile)
    return _CACHE["data"]


def recommend_for_user(users, courses, txns, user_id, top_n=5, level_filter=None, category_filter=None):
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


def fig_html(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=30, r=20, t=40, b=30))
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


# ---------------------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------------------
BASE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EduPro Learner Intelligence</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--blue:#2E5EAA;--bg:#0f1117;--panel:#171a23cc;--border:#262b38;--text:#e7e9ee;--muted:#9aa3b2;}
  *{box-sizing:border-box;}
  body{
    margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    color:var(--text);
    background-color:var(--bg);
    background-image:
      radial-gradient(circle at 15% 20%, rgba(46,94,170,0.16) 0%, transparent 45%),
      radial-gradient(circle at 85% 80%, rgba(21,128,61,0.14) 0%, transparent 45%),
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240' viewBox='0 0 240 240'%3E%3Cg fill='none' stroke='%232a3142' stroke-width='1.4' opacity='0.55'%3E%3Cpath d='M30 40 L30 60 L48 68 L66 60 L66 40 L48 32 Z'/%3E%3Cpath d='M30 40 L48 48 L66 40'/%3E%3Cpath d='M48 48 L48 68'/%3E%3Ccircle cx='150' cy='50' r='14'/%3E%3Cpath d='M150 36 L150 26 M142 22 L158 22'/%3E%3Crect x='180' y='150' width='28' height='20' rx='2'/%3E%3Cpath d='M180 150 L194 142 L208 150'/%3E%3Cpath d='M60 170 L60 200 M50 178 L70 178 M50 192 L70 192'/%3E%3Cpath d='M100 130 Q108 118 116 130 Q108 142 100 130 Z'/%3E%3C/g%3E%3C/svg%3E");
    background-repeat:no-repeat,no-repeat,repeat;
    background-attachment:fixed;
    min-height:100vh;
  }
  .topbar{display:flex;align-items:center;gap:24px;padding:14px 24px;background:var(--panel);backdrop-filter:blur(6px);border-bottom:1px solid var(--border);flex-wrap:wrap;position:sticky;top:0;z-index:10;}
  .topbar h1{font-size:18px;margin:0;white-space:nowrap;}
  nav{display:flex;gap:6px;flex-wrap:wrap;}
  nav a{color:var(--muted);text-decoration:none;padding:6px 12px;border-radius:8px;font-size:14px;}
  nav a.active{background:var(--blue);color:white;}
  nav a:hover{background:#22273a;color:var(--text);}
  .autoplay-bar{display:flex;align-items:center;gap:10px;margin-left:auto;font-size:13px;color:var(--muted);}
  .autoplay-bar button{background:#1c2130;border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:8px;cursor:pointer;font-size:13px;}
  .autoplay-bar button.on{background:var(--blue);border-color:var(--blue);color:white;}
  .progress-track{width:64px;height:4px;background:#262b38;border-radius:3px;overflow:hidden;}
  .progress-fill{height:100%;width:0%;background:var(--blue);}
  .wrap{padding:24px;max-width:1200px;margin:0 auto;}
  .grid{display:grid;gap:18px;}
  .cols-2{grid-template-columns:1fr 1fr;}
  .cols-3{grid-template-columns:1fr 1fr 1fr;}
  .cols-4{grid-template-columns:1fr 1fr 1fr 1fr;}
  @media (max-width:900px){.cols-2,.cols-3,.cols-4{grid-template-columns:1fr;}}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px;}
  .metric{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px 16px;}
  .metric .label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}
  .metric .value{font-size:26px;font-weight:700;margin-top:4px;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left;}
  th{color:var(--muted);font-weight:600;}
  .pill{display:inline-block;padding:4px 12px;border-radius:12px;color:white;font-weight:600;font-size:13px;}
  .info{background:#16324a;border:1px solid #2563eb55;border-radius:10px;padding:14px;color:#cfe3ff;}
  .warn{background:#3a2f12;border:1px solid #b45309;border-radius:10px;padding:14px;color:#fde8c2;}
  h2{font-size:20px;margin:28px 0 12px;}
  h3{font-size:16px;color:var(--muted);margin:0 0 10px;}
  select,input{background:#11141c;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:14px;}
  form.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:18px;}
  form.filters label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px;}
  button{background:var(--blue);color:white;border:none;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer;}
  .footer-note{color:var(--muted);font-size:12px;margin-top:30px;}
  .tabs{display:flex;gap:6px;margin-bottom:16px;}
  .tabs a{padding:8px 14px;border-radius:8px 8px 0 0;color:var(--muted);text-decoration:none;border:1px solid var(--border);border-bottom:none;font-size:14px;}
  .tabs a.active{background:var(--panel);color:var(--text);font-weight:600;}
</style>
</head>
<body>
<div class="topbar">
  <h1>🎓 EduPro</h1>
  <nav>
  {% for p in pages %}
    <a href="/?page={{ p|urlencode }}" class="{{ 'active' if p==page else '' }}">{{ p }}</a>
  {% endfor %}
  </nav>
  <div class="autoplay-bar">
    <span id="autoplayLabel">Auto-advance: off</span>
    <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
    <button id="autoplayBtn" type="button">▶ Play</button>
  </div>
</div>
<div class="wrap">
  {{ body|safe }}
  <div class="footer-note">Learners: {{ n_users }} | Courses: {{ n_courses }} | Transactions: {{ n_txns }}</div>
</div>
<script>
(function(){
  var PAGES = {{ pages|tojson }};
  var CURRENT = {{ page|tojson }};
  var DELAY_MS = 12000; // 12s per page before auto-advancing
  var STEP_MS = 100;

  var btn = document.getElementById("autoplayBtn");
  var label = document.getElementById("autoplayLabel");
  var fill = document.getElementById("progressFill");

  var enabled = localStorage.getItem("edupro_autoplay") === "1";
  var elapsed = 0;
  var timerId = null;

  function nextPageUrl(){
    var idx = PAGES.indexOf(CURRENT);
    var nextIdx = (idx + 1) % PAGES.length;
    return "/?page=" + encodeURIComponent(PAGES[nextIdx]) + "&auto=1";
  }

  function updateUI(){
    btn.textContent = enabled ? "⏸ Pause" : "▶ Play";
    btn.className = enabled ? "on" : "";
    label.textContent = "Auto-advance: " + (enabled ? "on" : "off");
  }

  function tick(){
    elapsed += STEP_MS;
    var pct = Math.min(100, (elapsed / DELAY_MS) * 100);
    fill.style.width = pct + "%";
    if (elapsed >= DELAY_MS){
      window.location.href = nextPageUrl();
    }
  }

  function start(){
    elapsed = 0;
    fill.style.width = "0%";
    if (timerId) clearInterval(timerId);
    timerId = setInterval(tick, STEP_MS);
  }

  function stop(){
    if (timerId) clearInterval(timerId);
    timerId = null;
    fill.style.width = "0%";
  }

  btn.addEventListener("click", function(){
    enabled = !enabled;
    localStorage.setItem("edupro_autoplay", enabled ? "1" : "0");
    updateUI();
    if (enabled) start(); else stop();
  });

  updateUI();
  if (enabled) start();
})();
</script>
</body>
</html>
"""


def render(body_html, page):
    users, courses, txns, *_ = load_data()
    return render_template_string(
        BASE, body=body_html, page=page, pages=PAGES,
        n_users=f"{len(users):,}", n_courses=f"{len(courses):,}", n_txns=f"{len(txns):,}",
    )


# ---------------------------------------------------------------------
# PAGE BUILDERS
# ---------------------------------------------------------------------
def page_overview():
    users, courses, txns, eval_summary, cluster_meta, _ = load_data()

    seg_counts = users[users.Cluster != -1]["SegmentName"].value_counts().reindex(
        [n for n in CLUSTER_NAMES.values() if n != "Unsegmented (New/Inactive)"]
    )
    fig1 = px.bar(x=seg_counts.index, y=seg_counts.values, color=seg_counts.index,
                  color_discrete_map=PALETTE, labels={"x": "Segment", "y": "Number of Learners"})
    fig1.update_layout(showlegend=False)

    tc = txns.merge(courses, on="CourseID")
    cat_counts = tc["CourseCategory"].value_counts()
    fig2 = px.bar(x=cat_counts.values, y=cat_counts.index, orientation="h",
                  labels={"x": "Enrollments", "y": ""}, color_discrete_sequence=["#2E5EAA"])

    trend = txns.set_index("TransactionDate").resample("MS").size().reset_index(name="Enrollments")
    fig3 = px.line(trend, x="TransactionDate", y="Enrollments", markers=True,
                   color_discrete_sequence=["#15803D"])

    html = f"""
    <h2>EduPro Student Segmentation & Personalized Recommendation System</h2>
    <p>A data-driven personalization engine that segments learners into behavioral cohorts
    and serves cluster-aware, content-based course recommendations.</p>
    <div class="grid cols-4">
      <div class="metric"><div class="label">Total Learners</div><div class="value">{len(users):,}</div></div>
      <div class="metric"><div class="label">Total Courses</div><div class="value">{len(courses):,}</div></div>
      <div class="metric"><div class="label">Total Transactions</div><div class="value">{len(txns):,}</div></div>
      <div class="metric"><div class="label">Segments Identified</div><div class="value">{cluster_meta['final_k']}</div></div>
    </div>
    <h3 style="margin-top:24px">Segment Distribution</h3>
    <div class="card">{fig_html(fig1)}</div>
    <div class="grid cols-2" style="margin-top:18px">
      <div class="card"><h3>Enrollments by Category</h3>{fig_html(fig2)}</div>
      <div class="card"><h3>Monthly Enrollment Trend</h3>{fig_html(fig3)}</div>
    </div>
    <h3 style="margin-top:24px">Key Insight</h3>
    <div class="info">
      Five behaviorally distinct learner segments emerge from enrollment, spend, and depth patterns.
      Cluster-aware recommendations lift the average rating of suggested courses relative to platform
      baseline by <b>{eval_summary['engagement_lift_pct_proxy']:.1f}%</b> (proxy estimate), with
      <b>Precision@5 of {eval_summary['recommendation_precision_at_5']:.1%}</b> on held-out enrollments.
    </div>
    """
    return html


def page_learner_explorer():
    users, courses, txns, *_ = load_data()
    search = request.args.get("search", "")
    filtered = users[users.UserID.str.contains(search, case=False)] if search else users
    options = filtered["UserID"].head(500).tolist()
    user_id = request.args.get("user_id") or (options[0] if options else None)

    opt_html = "".join(
        f'<option value="{u}" {"selected" if u == user_id else ""}>{u}</option>' for u in options
    )

    if not user_id or users[users.UserID == user_id].empty:
        return f"""
        <h2>Learner Profile Explorer</h2>
        <form class="filters" method="get">
          <input type="hidden" name="page" value="Learner Profile Explorer">
          <div><label>Search by UserID</label><input name="search" value="{search}" placeholder="e.g. U00001"></div>
          <button type="submit">Search</button>
        </form>
        <div class="warn">No learner found.</div>
        """

    urow = users[users.UserID == user_id].iloc[0]
    seg_name = urow["SegmentName"]
    color = PALETTE.get(seg_name, "#999")

    hist = txns[txns.UserID == user_id].merge(courses, on="CourseID").sort_values("TransactionDate", ascending=False)
    if hist.empty:
        hist_html = '<div class="warn">This learner has no transaction history yet.</div>'
    else:
        rows = "".join(
            f"<tr><td>{r.TransactionDate.date()}</td><td>{r.CourseID}</td><td>{r.CourseCategory}</td>"
            f"<td>{r.CourseType}</td><td>{r.CourseLevel}</td><td>{r.CourseRating}</td><td>${r.Amount:.0f}</td></tr>"
            for r in hist.itertuples()
        )
        hist_html = f"""<table><thead><tr><th>Date</th><th>Course</th><th>Category</th><th>Type</th>
        <th>Level</th><th>Rating</th><th>Amount</th></tr></thead><tbody>{rows}</tbody></table>"""

    feats = ["TotalCoursesEnrolled", "NumCategoriesExplored", "AvgSpendPerLearner", "LearningDepthIndex", "CertificationShare"]
    seg_avg = users[users.Cluster == urow["Cluster"]][feats].mean()
    norm_max = users[feats].max()
    radar_user = [urow[f] / norm_max[f] if norm_max[f] else 0 for f in feats]
    radar_seg = [seg_avg[f] / norm_max[f] if norm_max[f] else 0 for f in feats]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=radar_user + [radar_user[0]], theta=feats + [feats[0]], fill="toself", name="This Learner"))
    fig.add_trace(go.Scatterpolar(r=radar_seg + [radar_seg[0]], theta=feats + [feats[0]], fill="toself", name=f"{seg_name} Avg", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))

    return f"""
    <h2>Learner Profile Explorer</h2>
    <p>Select a learner to view their full profile, behavioral features, and assigned segment.</p>
    <form class="filters" method="get">
      <input type="hidden" name="page" value="Learner Profile Explorer">
      <div><label>Search by UserID</label><input name="search" value="{search}" placeholder="e.g. U00001"></div>
      <div><label>Select Learner</label><select name="user_id">{opt_html}</select></div>
      <button type="submit">Go</button>
    </form>

    <div class="grid cols-3">
      <div class="card"><h3>Demographics</h3>
        <p><b>UserID:</b> {urow['UserID']}<br><b>Age:</b> {int(urow['Age'])}<br><b>Gender:</b> {urow['Gender']}</p>
      </div>
      <div class="card"><h3>Segment</h3>
        <span class="pill" style="background:{color}">{seg_name}</span>
        <p style="color:var(--muted);margin-top:8px">{CLUSTER_DESCRIPTIONS.get(seg_name,'')}</p>
      </div>
      <div class="card"><h3>Behavioral Snapshot</h3>
        <div class="grid cols-3">
          <div class="metric"><div class="label">Courses Enrolled</div><div class="value">{int(urow['TotalCoursesEnrolled'])}</div></div>
          <div class="metric"><div class="label">Categories Explored</div><div class="value">{int(urow['NumCategoriesExplored'])}</div></div>
          <div class="metric"><div class="label">Avg Spend</div><div class="value">${urow['AvgSpendPerLearner']:.0f}</div></div>
        </div>
        <div class="grid cols-3" style="margin-top:10px">
          <div class="metric"><div class="label">Learning Depth</div><div class="value">{urow['LearningDepthIndex']:.2f}</div></div>
          <div class="metric"><div class="label">Pref. Category</div><div class="value" style="font-size:16px">{urow['PreferredCategory']}</div></div>
          <div class="metric"><div class="label">Pref. Level</div><div class="value" style="font-size:16px">{urow['PreferredLevel']}</div></div>
        </div>
      </div>
    </div>

    <h3 style="margin-top:24px">Enrollment History</h3>
    <div class="card">{hist_html}</div>

    <h3 style="margin-top:24px">Feature Radar (vs. Segment Average)</h3>
    <div class="card">{fig_html(fig, height=460)}</div>
    """


def page_cluster_viz():
    users, courses, txns, eval_summary, cluster_meta, cluster_profile = load_data()
    tab = request.args.get("tab", "pca")

    tabs_html = f"""
    <div class="tabs">
      <a href="/?page=Cluster+Visualization&tab=pca" class="{'active' if tab=='pca' else ''}">PCA Scatter</a>
      <a href="/?page=Cluster+Visualization&tab=elbow" class="{'active' if tab=='elbow' else ''}">Elbow & Silhouette</a>
      <a href="/?page=Cluster+Visualization&tab=profiles" class="{'active' if tab=='profiles' else ''}">Cluster Profiles</a>
    </div>
    """

    if tab == "elbow":
        sil_by_k = cluster_meta["silhouette_by_k"]
        ks = list(sil_by_k.keys())
        sils = list(sil_by_k.values())
        fig = px.line(x=ks, y=sils, markers=True, labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"})
        fig.add_vline(x=cluster_meta["final_k"], line_dash="dash", line_color="red",
                      annotation_text=f"Selected K={cluster_meta['final_k']}")
        body = f"""<div class="card">{fig_html(fig, height=460)}</div>
        <p style="color:var(--muted);margin-top:10px">
        Final model: K={cluster_meta['final_k']}, Silhouette={cluster_meta['final_silhouette']:.3f}.
        K=5 was selected over the silhouette-maximizing K=3 to preserve business-actionable segment
        granularity (3 clusters collapsed Career-Switchers and Premium Upskillers together, losing
        monetization-relevant distinctions).</p>"""
    elif tab == "profiles":
        display_profile = cluster_profile.copy()
        display_profile["Cluster"] = display_profile["Cluster"].map(CLUSTER_NAMES)
        body = f'<div class="card">{display_profile.to_html(index=False, border=0, classes="dataframe")}</div>'
    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        plot_df = users[users.Cluster != -1].copy()
        feats = ["Age", "TotalCoursesEnrolled", "NumCategoriesExplored", "AvgCoursesPerCategory",
                 "EnrollmentFrequency", "AvgCourseRatingEnrolled", "AvgSpendPerLearner",
                 "DiversityScore", "LearningDepthIndex", "BeginnerShare", "AdvancedShare", "CertificationShare"]
        X = StandardScaler().fit_transform(plot_df[feats])
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        plot_df["PCA1"], plot_df["PCA2"] = coords[:, 0], coords[:, 1]
        fig = px.scatter(plot_df, x="PCA1", y="PCA2", color="SegmentName",
                          color_discrete_map=PALETTE, opacity=0.65,
                          hover_data=["UserID", "TotalCoursesEnrolled", "PreferredCategory"])
        body = f'<div class="card">{fig_html(fig, height=560)}</div>'

    return f"<h2>Cluster Visualization Dashboard</h2>{tabs_html}{body}"


def page_recommendations():
    users, courses, txns, *_ = load_data()
    user_options = users["UserID"].head(500).tolist()
    user_id = request.args.get("user_id") or user_options[0]
    level_filter = request.args.get("level", "Any")
    cat_filter = request.args.get("category", "Any")
    top_n = int(request.args.get("top_n", 5))

    cat_options = ["Any"] + sorted(courses["CourseCategory"].unique().tolist())

    user_opt_html = "".join(f'<option value="{u}" {"selected" if u==user_id else ""}>{u}</option>' for u in user_options)
    level_opt_html = "".join(
        f'<option value="{l}" {"selected" if l==level_filter else ""}>{l}</option>'
        for l in ["Any", "Beginner", "Intermediate", "Advanced"]
    )
    cat_opt_html = "".join(f'<option value="{c}" {"selected" if c==cat_filter else ""}>{c}</option>' for c in cat_options)

    urow = users[users.UserID == user_id].iloc[0]
    seg_name = urow["SegmentName"]
    color = PALETTE.get(seg_name, "#999")

    recs = recommend_for_user(users, courses, txns, user_id, top_n=top_n, level_filter=level_filter, category_filter=cat_filter)

    if recs.empty:
        recs_html = '<div class="warn">No recommendations available for this filter combination.</div>'
    else:
        cards = "".join(
            f"""<div class="card" style="margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
                <div><b>{r.CourseID}</b> — {r.CourseCategory}<br>
                <span style="color:var(--muted);font-size:13px">{r.CourseType} · {r.CourseLevel}</span></div>
                <div class="metric" style="min-width:110px"><div class="label">Rating</div><div class="value">{r.CourseRating:.1f} ⭐</div></div>
                <div class="metric" style="min-width:110px"><div class="label">Relevance</div><div class="value">{r.RelevanceScore:.2f}</div></div>
            </div>"""
            for r in recs.itertuples()
        )
        table_html = recs.to_html(index=False, border=0, classes="dataframe")
        recs_html = f"""
        <h3 style="margin-top:24px">Recommended Courses</h3>
        {cards}
        <h3 style="margin-top:24px">Why these courses? (Scoring Breakdown)</h3>
        <p style="color:var(--muted)">Relevance Score = 30% Category Match + 15% Level Match + 15% Level Proximity
        + 20% Cluster Popularity + 20% Rating-Weighted Relevance</p>
        <div class="card">{table_html}</div>
        """

    return f"""
    <h2>Personalized Course Recommendations</h2>
    <form class="filters" method="get">
      <input type="hidden" name="page" value="Personalized Recommendations">
      <div><label>Learner</label><select name="user_id">{user_opt_html}</select></div>
      <div><label>Filter by Level</label><select name="level">{level_opt_html}</select></div>
      <div><label>Filter by Category</label><select name="category">{cat_opt_html}</select></div>
      <div><label>Number of Recommendations</label>
        <input type="number" name="top_n" min="3" max="10" value="{top_n}" style="width:80px"></div>
      <button type="submit">Apply</button>
    </form>
    <p><b>Segment:</b> <span class="pill" style="background:{color}">{seg_name}</span>
    &nbsp;&nbsp; <b>Preferred Category:</b> {urow['PreferredCategory']}
    &nbsp;&nbsp; <b>Preferred Level:</b> {urow['PreferredLevel']}</p>
    {recs_html}
    """


def page_segment_comparison():
    users, courses, txns, *_ = load_data()
    all_segs = [n for n in CLUSTER_NAMES.values() if n != "Unsegmented (New/Inactive)"]
    segs = request.args.getlist("segs") or all_segs
    feats = ["TotalCoursesEnrolled", "AvgSpendPerLearner", "DiversityScore",
             "LearningDepthIndex", "CertificationShare", "AvgCourseRatingEnrolled"]
    metric = request.args.get("metric", feats[0])

    seg_checkboxes = "".join(
        f'<label style="margin-right:14px;font-size:13px;color:var(--text)">'
        f'<input type="checkbox" name="segs" value="{s}" {"checked" if s in segs else ""}> {s}</label>'
        for s in all_segs
    )
    metric_opt_html = "".join(f'<option value="{m}" {"selected" if m==metric else ""}>{m}</option>' for m in feats)

    comp_df = users[users.SegmentName.isin(segs)]
    fig = px.box(comp_df, x="SegmentName", y=metric, color="SegmentName", color_discrete_map=PALETTE,
                 category_orders={"SegmentName": segs})
    fig.update_layout(showlegend=False)

    summary = comp_df.groupby("SegmentName")[feats].mean().round(2)
    summary["N_Learners"] = comp_df.groupby("SegmentName").size()

    top_cat = comp_df.groupby("SegmentName")["PreferredCategory"].agg(lambda x: x.value_counts().idxmax())
    top_lvl = comp_df.groupby("SegmentName")["PreferredLevel"].agg(lambda x: x.value_counts().idxmax())
    top_table = pd.DataFrame({"Top Category": top_cat, "Top Level": top_lvl})

    profiles_html = "".join(
        f"<p><b>{s}:</b> {CLUSTER_DESCRIPTIONS.get(s,'')}</p>" for s in segs
    )

    return f"""
    <h2>Segment Comparison Panel</h2>
    <form class="filters" method="get" style="align-items:start">
      <input type="hidden" name="page" value="Segment Comparison">
      <div><label>Select segments to compare</label><div>{seg_checkboxes}</div></div>
      <div><label>Metric to Compare</label><select name="metric">{metric_opt_html}</select></div>
      <button type="submit">Apply</button>
    </form>
    <div class="card">{fig_html(fig, height=480)}</div>

    <h3 style="margin-top:24px">Segment Summary Table</h3>
    <div class="card">{summary.to_html(border=0, classes="dataframe")}</div>

    <h3 style="margin-top:24px">Top Preferred Category by Segment</h3>
    <div class="card">{top_table.to_html(border=0, classes="dataframe")}</div>

    <h3 style="margin-top:24px">Segment Profiles</h3>
    {profiles_html}
    """


def page_model_evaluation():
    users, courses, txns, eval_summary, cluster_meta, cluster_profile = load_data()

    intra = eval_summary["intra_cluster_similarity"]
    intra_named = {CLUSTER_NAMES[int(k)]: v for k, v in intra.items()}
    fig = px.bar(x=list(intra_named.keys()), y=list(intra_named.values()),
                 labels={"x": "Segment", "y": "Avg. Cosine Similarity (within cluster)"},
                 color=list(intra_named.keys()), color_discrete_map=PALETTE)
    fig.add_hline(y=eval_summary["avg_cross_cluster_similarity"], line_dash="dash", line_color="red",
                  annotation_text="Cross-cluster baseline")
    fig.update_layout(showlegend=False)

    ref_table = pd.DataFrame({
        "Metric": ["Silhouette Score", "Intra-Cluster Similarity", "Recommendation Precision", "Engagement Lift (Proxy)"],
        "Purpose": ["Cluster quality / separation", "Behavioral consistency within segments",
                    "Relevance of top-N recommendations", "Estimated impact of personalization"],
        "Result": [f"{eval_summary['silhouette_score_final']:.3f}",
                   "0.18 – 0.69 across segments (vs. -0.11 baseline)",
                   f"{eval_summary['recommendation_precision_at_5']:.1%} @ Top-5",
                   f"+{eval_summary['engagement_lift_pct_proxy']:.1f}% avg rating vs. baseline"],
    })

    return f"""
    <h2>Evaluation & Validation</h2>
    <div class="grid cols-4">
      <div class="metric"><div class="label">Silhouette (K-Means)</div><div class="value">{eval_summary['silhouette_score_final']:.3f}</div></div>
      <div class="metric"><div class="label">Hierarchical Silhouette</div><div class="value">{eval_summary['hierarchical_silhouette']:.3f}</div></div>
      <div class="metric"><div class="label">Precision@5</div><div class="value">{eval_summary['recommendation_precision_at_5']:.1%}</div></div>
      <div class="metric"><div class="label">Engagement Lift</div><div class="value">{eval_summary['engagement_lift_pct_proxy']:.1f}%</div></div>
    </div>

    <h3 style="margin-top:24px">Cross-Validation: K-Means vs. Hierarchical Clustering</h3>
    <p>Adjusted Rand Index between K-Means and Agglomerative (Ward) clustering:
    <b>{eval_summary['ari_kmeans_vs_hierarchical']:.3f}</b> — indicates substantial agreement
    between the two independent algorithms on learner groupings, validating that segments
    reflect real structure rather than algorithm-specific artifacts.</p>

    <h3 style="margin-top:24px">Intra-Cluster Similarity (Behavioral Consistency)</h3>
    <div class="card">{fig_html(fig, height=440)}</div>
    <p style="color:var(--muted)">Cross-cluster (control) similarity averages {eval_summary['avg_cross_cluster_similarity']:.3f}.
    All segments show higher within-cluster similarity than the random cross-cluster baseline,
    confirming behaviorally coherent groupings.</p>

    <h3 style="margin-top:24px">Metric Reference</h3>
    <div class="card">{ref_table.to_html(index=False, border=0, classes="dataframe")}</div>
    """


PAGE_BUILDERS = {
    "Overview": page_overview,
    "Learner Profile Explorer": page_learner_explorer,
    "Cluster Visualization": page_cluster_viz,
    "Personalized Recommendations": page_recommendations,
    "Segment Comparison": page_segment_comparison,
    "Model Evaluation": page_model_evaluation,
}


@app.route("/")
def index():
    page = request.args.get("page", "Overview")
    if page not in PAGE_BUILDERS:
        page = "Overview"
    body = PAGE_BUILDERS[page]()
    return render(body, page)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
