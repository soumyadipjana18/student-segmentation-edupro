import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA = os.path.join(_ROOT, "data")
OUT = os.path.join(_ROOT, "outputs")
sns.set_style("whitegrid")

users = pd.read_csv(f"{DATA}/users.csv")
courses = pd.read_csv(f"{DATA}/courses.csv")
txns = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["TransactionDate"])
learners = pd.read_csv(f"{DATA}/learner_profiles_clustered.csv")

CLUSTER_NAMES = {0: "Specialists", 1: "Explorers", 2: "Premium Upskillers",
                  3: "Casual Dabblers", 4: "Niche Beginners"}
learners["SegmentName"] = learners["Cluster"].map(CLUSTER_NAMES).fillna("Unsegmented")

# 1. Category popularity
plt.figure(figsize=(8, 4.5))
tc = txns.merge(courses, on="CourseID")
order = tc["CourseCategory"].value_counts().index
sns.countplot(data=tc, y="CourseCategory", order=order, color="#2E5EAA")
plt.title("Course Enrollments by Category")
plt.xlabel("Number of Enrollments")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_category_popularity.png", dpi=150)
plt.close()

# 2. Course level distribution
plt.figure(figsize=(6, 4.5))
sns.countplot(data=tc, x="CourseLevel", order=["Beginner", "Intermediate", "Advanced"], color="#C2410C")
plt.title("Enrollments by Course Level")
plt.ylabel("Number of Enrollments")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_level_distribution.png", dpi=150)
plt.close()

# 3. Segment sizes
plt.figure(figsize=(7, 4.5))
seg_counts = learners[learners.Cluster != -1]["SegmentName"].value_counts()
palette = ["#2E5EAA", "#C2410C", "#15803D", "#7C3AED", "#DB2777"]
plt.bar(seg_counts.index, seg_counts.values, color=palette[:len(seg_counts)])
plt.title("Learner Segment Sizes")
plt.ylabel("Number of Learners")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/segment_sizes.png", dpi=150)
plt.close()

# 4. Spend distribution by segment
plt.figure(figsize=(7.5, 4.5))
order_seg = ["Specialists", "Explorers", "Premium Upskillers", "Casual Dabblers", "Niche Beginners"]
sns.boxplot(data=learners[learners.Cluster != -1], x="SegmentName", y="AvgSpendPerLearner",
            order=order_seg, palette=palette)
plt.title("Average Spend per Learner by Segment")
plt.xlabel("")
plt.ylabel("Avg Spend ($)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/segment_spend_boxplot.png", dpi=150)
plt.close()

# 5. Diversity vs Depth scatter colored by segment
plt.figure(figsize=(7, 6))
for i, seg in enumerate(order_seg):
    sub = learners[learners.SegmentName == seg]
    plt.scatter(sub["DiversityScore"], sub["LearningDepthIndex"], s=20, alpha=0.6,
                color=palette[i], label=seg)
plt.title("Diversity Score vs Learning Depth Index by Segment")
plt.xlabel("Diversity Score (Categories Explored)")
plt.ylabel("Learning Depth Index")
plt.legend(fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig(f"{OUT}/diversity_vs_depth.png", dpi=150)
plt.close()

# 6. Age distribution overall
plt.figure(figsize=(6.5, 4.5))
sns.histplot(users["Age"], bins=25, color="#2E5EAA", kde=True)
plt.title("Learner Age Distribution")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(f"{OUT}/age_distribution.png", dpi=150)
plt.close()

# 7. Monthly enrollment trend
plt.figure(figsize=(8.5, 4.5))
trend = txns.set_index("TransactionDate").resample("MS").size()
plt.plot(trend.index, trend.values, marker="o", color="#15803D", linewidth=1.5)
plt.title("Monthly Enrollment Volume Over Time")
plt.ylabel("Enrollments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/monthly_trend.png", dpi=150)
plt.close()

print("EDA visuals generated.")
