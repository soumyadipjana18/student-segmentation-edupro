# EduPro — Student Segmentation & Personalized Course Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Clustering-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

A data-driven personalization engine for EduPro's online learning platform. The system
segments learners into behaviorally distinct cohorts using unsupervised clustering and
serves cluster-aware, content-based course recommendations through an interactive
Streamlit dashboard.

> Internship project — Unified Mentor × Toronto Government Parks, Forestry & Recreation

---

## Problem Statement

EduPro currently sends the same course recommendations to every learner, regardless of
their goals, behavior, or experience level. This one-size-fits-all approach:

- Fails to maximize learner engagement
- Limits course completion rates
- Leaves retention and platform loyalty opportunities on the table

This project replaces that generic approach with a **structured learner segmentation
framework** and a **personalized recommendation engine** built on top of it.

## What This Project Delivers

| Deliverable | Location |
|---|---|
| Full data science pipeline (feature engineering → clustering → recommendation → evaluation) | [`src/`](src/) |
| Interactive Streamlit dashboard | [`app.py`](app.py) |
| Research paper (EDA, methodology, results) | [`docs/EduPro_Research_Paper.pdf`](docs/EduPro_Research_Paper.pdf) |
| Executive summary for stakeholders | [`docs/EduPro_Executive_Summary.pdf`](docs/EduPro_Executive_Summary.pdf) |
| Generated charts, metrics, cluster summaries | [`outputs/`](outputs/) |

---

## Architecture

```
Raw Data (Users / Courses / Transactions)
        │
        ▼
[src/01_generate_data.py]          synthetic data generator (swap for real EduPro extracts)
        │
        ▼
[src/02_feature_engineering.py]    learner-level aggregation: engagement, preference,
        │                          and behavioral features
        ▼
[src/03_clustering.py]             preprocessing + K-Means (elbow/silhouette) +
        │                          Hierarchical validation + PCA visualization
        ▼
[src/04_recommendation_engine.py]  cluster-aware, content-based, rating-weighted
        │                          recommendation logic + precision evaluation
        ▼
[src/05_evaluation.py]             intra-cluster similarity, cross-cluster baseline,
        │                          consolidated evaluation metrics
        ▼
[src/06_eda_visuals.py]            chart generation for the research paper
        │
        ▼
[app.py]                           Streamlit dashboard reading all of the above
```

## Discovered Learner Segments

| Segment | Share | Defining Behavior |
|---|---|---|
| **Specialists** | 33% | Deep focus within 1–2 categories, intermediate level |
| **Explorers** | 22% | Widest spread across categories, highest course count |
| **Premium Upskillers** | 23% | Lower volume, highest spend, advanced/certification-focused |
| **Casual Dabblers** | 14% | Lowest engagement and spend — churn risk |
| **Niche Beginners** | 7% | Few courses in one niche, high certification share |

Validated with two independent clustering algorithms (K-Means + Hierarchical/Ward),
Adjusted Rand Index = 0.601, and intra-cluster cosine similarity of 0.18–0.69 vs. a
-0.11 cross-cluster baseline.

## Key Results

| Metric | Result |
|---|---|
| Silhouette Score (K-Means, K=5) | 0.197 |
| Hierarchical Silhouette (validation) | 0.182 |
| Recommendation Precision@5 | 62.3% |
| Engagement Lift (proxy) | +3.5% avg. rating vs. baseline |

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/edupro-segmentation.git
cd edupro-segmentation
pip install -r requirements.txt
```

### 2. Run the full pipeline (optional — pre-computed outputs are already included)

```bash
bash run_pipeline.sh
```

This regenerates everything in `data/` and `outputs/` from scratch, in order:
data generation → feature engineering → clustering → recommendation engine →
evaluation → EDA visuals.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens with six modules: **Overview**, **Learner Profile Explorer**,
**Cluster Visualization**, **Personalized Recommendations**, **Segment Comparison**,
and **Model Evaluation**.

---

## Using Real EduPro Data

Replace the three files in `data/` with real extracts using the same schema:

- **Users** — `UserID`, `Age`, `Gender`
- **Courses** — `CourseID`, `CourseCategory`, `CourseType`, `CourseLevel`, `CourseRating`
- **Transactions** — `UserID`, `CourseID`, `TransactionDate`, `Amount`

Then re-run `src/02_feature_engineering.py` through `src/06_eda_visuals.py` (skip
`01_generate_data.py`, which only exists to produce synthetic placeholder data).
No changes to `app.py` are required.

---

## Methodology Summary

1. **Feature Engineering** — Engagement (courses enrolled, courses/category,
   enrollment frequency), Preference (preferred category/level, avg. rating enrolled),
   and Behavioral (spend, diversity score, learning depth index, certification share)
   features aggregated per learner.
2. **Preprocessing** — Numeric features winsorized (1st/99th percentile) and
   standardized; categorical features one-hot encoded.
3. **Segmentation** — K-Means (K chosen via elbow + silhouette, with a business-driven
   override to K=5 for actionable granularity), cross-validated against Agglomerative
   (Ward) hierarchical clustering via Adjusted Rand Index.
4. **Recommendation Engine** — Weighted score: 30% category match + 15% level match +
   15% level proximity + 20% in-cluster course popularity + 20% rating-weighted relevance.
5. **Evaluation** — Silhouette score, intra-cluster cosine similarity vs. cross-cluster
   baseline, held-out category-match Precision@5, and an engagement-lift proxy.

Full methodology, charts, and discussion are in
[`docs/EduPro_Research_Paper.pdf`](docs/EduPro_Research_Paper.pdf).

## Project Structure

```
edupro-segmentation/
├── src/
│   ├── 01_generate_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_clustering.py
│   ├── 04_recommendation_engine.py
│   ├── 05_evaluation.py
│   └── 06_eda_visuals.py
├── data/                      # Users / Courses / Transactions + engineered profiles
├── outputs/                   # Generated charts, JSON metrics, cluster summaries
├── docs/                      # Research paper & executive summary (PDF)
├── app.py                     # Streamlit dashboard
├── run_pipeline.sh            # Runs the full pipeline in order
├── requirements.txt
├── .streamlit/config.toml     # Dashboard theme
└── README.md
```

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Streamlit + Plotly** — interactive dashboard
- **K-Means / Agglomerative Clustering** — unsupervised segmentation
- **PCA** — cluster visualization

## License

MIT — see [LICENSE](LICENSE).
