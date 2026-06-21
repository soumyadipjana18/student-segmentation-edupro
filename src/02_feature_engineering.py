"""
Feature Engineering: Learner-Level Aggregation
Builds one row per learner (UserID) combining demographics + behavior.
"""

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

import pandas as pd
import numpy as np

DATA = os.path.join(_ROOT, "data")

users = pd.read_csv(f"{DATA}/users.csv")
courses = pd.read_csv(f"{DATA}/courses.csv")
txns = pd.read_csv(f"{DATA}/transactions.csv", parse_dates=["TransactionDate"])

# Join transactions with course metadata
tc = txns.merge(courses, on="CourseID", how="left")
tc = tc.merge(users, on="UserID", how="left")

level_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
tc["LevelScore"] = tc["CourseLevel"].map(level_map)

profiles = []
for uid, g in tc.groupby("UserID"):
    n_courses = g["CourseID"].nunique()
    n_categories = g["CourseCategory"].nunique()
    avg_per_category = n_courses / max(n_categories, 1)

    span_days = (g["TransactionDate"].max() - g["TransactionDate"].min()).days
    enrollment_frequency = n_courses / (span_days / 30 + 1)  # courses per month, smoothed

    # Preference features
    preferred_category = g["CourseCategory"].value_counts().idxmax()
    preferred_level = g["CourseLevel"].value_counts().idxmax()
    avg_rating = g["CourseRating"].mean()

    # Behavioral features
    avg_spend = g["Amount"].mean()
    total_spend = g["Amount"].sum()
    diversity_score = n_categories  # number of distinct categories explored
    learning_depth_index = (g["LevelScore"] >= 1).mean()  # share of intermediate+advanced
    beginner_share = (g["LevelScore"] == 0).mean()
    advanced_share = (g["LevelScore"] == 2).mean()

    cert_share = (g["CourseType"] == "Certification").mean()

    profiles.append({
        "UserID": uid,
        "TotalCoursesEnrolled": n_courses,
        "NumCategoriesExplored": n_categories,
        "AvgCoursesPerCategory": avg_per_category,
        "EnrollmentFrequency": enrollment_frequency,
        "PreferredCategory": preferred_category,
        "PreferredLevel": preferred_level,
        "AvgCourseRatingEnrolled": avg_rating,
        "AvgSpendPerLearner": avg_spend,
        "TotalSpend": total_spend,
        "DiversityScore": diversity_score,
        "LearningDepthIndex": learning_depth_index,
        "BeginnerShare": beginner_share,
        "AdvancedShare": advanced_share,
        "CertificationShare": cert_share,
        "ActiveSpanDays": span_days,
    })

profiles_df = pd.DataFrame(profiles)
learner_df = users.merge(profiles_df, on="UserID", how="left")

# Users with zero transactions (if any) get filled with neutral defaults
learner_df = learner_df.fillna({
    "TotalCoursesEnrolled": 0, "NumCategoriesExplored": 0, "AvgCoursesPerCategory": 0,
    "EnrollmentFrequency": 0, "AvgCourseRatingEnrolled": 0, "AvgSpendPerLearner": 0,
    "TotalSpend": 0, "DiversityScore": 0, "LearningDepthIndex": 0,
    "BeginnerShare": 0, "AdvancedShare": 0, "CertificationShare": 0, "ActiveSpanDays": 0
})
learner_df["PreferredCategory"] = learner_df["PreferredCategory"].fillna("None")
learner_df["PreferredLevel"] = learner_df["PreferredLevel"].fillna("None")

learner_df.to_csv(f"{DATA}/learner_profiles.csv", index=False)
print("Learner profile shape:", learner_df.shape)
print(learner_df.describe(include="all").T)
