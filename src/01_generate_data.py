"""
Generate a realistic synthetic dataset for EduPro:
- Users sheet
- Courses sheet
- Transactions sheet

The generation deliberately embeds 4-5 latent learner archetypes
(explorer, specialist, career-switcher, casual dabbler, premium upskiller)
so that downstream clustering has real, recoverable structure -- mirroring
how a real online-learning platform's behavioral data would look.
"""

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_ROOT, "data")
os.makedirs(_DATA_DIR, exist_ok=True)

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. COURSES
# ---------------------------------------------------------------
categories = [
    "Data Science", "Web Development", "Business", "Design",
    "Marketing", "Cloud Computing", "Cybersecurity", "Personal Development"
]
course_types = ["Self-Paced", "Instructor-Led", "Certification", "Workshop"]
levels = ["Beginner", "Intermediate", "Advanced"]

N_COURSES = 180
course_rows = []
cid = 1
for _ in range(N_COURSES):
    cat = rng.choice(categories, p=[0.20, 0.18, 0.14, 0.10, 0.12, 0.10, 0.08, 0.08])
    level = rng.choice(levels, p=[0.45, 0.35, 0.20])
    ctype = rng.choice(course_types, p=[0.45, 0.25, 0.20, 0.10])
    # rating slightly correlated with type (certifications/instructor-led rated a bit higher)
    base = 3.6
    if ctype == "Certification":
        base += 0.35
    if ctype == "Instructor-Led":
        base += 0.15
    rating = np.clip(rng.normal(base, 0.35), 2.5, 5.0)
    course_rows.append([f"C{cid:04d}", cat, ctype, level, round(rating, 1)])
    cid += 1

courses = pd.DataFrame(course_rows, columns=[
    "CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating"
])

# price depends on type/level (used to simulate transaction Amount)
def base_price(ctype, level):
    p = {"Self-Paced": 25, "Workshop": 40, "Instructor-Led": 70, "Certification": 120}[ctype]
    p += {"Beginner": 0, "Intermediate": 15, "Advanced": 35}[level]
    return p

courses["BasePrice"] = courses.apply(lambda r: base_price(r.CourseType, r.CourseLevel), axis=1)

# ---------------------------------------------------------------
# 2. USERS  (with latent archetype, hidden from downstream analysis)
# ---------------------------------------------------------------
N_USERS = 1500
archetypes = ["Explorer", "Specialist", "Career-Switcher", "Casual Dabbler", "Premium Upskiller"]
archetype_p = [0.22, 0.22, 0.20, 0.21, 0.15]

user_rows = []
for uid in range(1, N_USERS + 1):
    arche = rng.choice(archetypes, p=archetype_p)
    age = int(np.clip(rng.normal(29, 8), 16, 65))
    gender = rng.choice(["Male", "Female", "Other"], p=[0.52, 0.45, 0.03])
    user_rows.append([f"U{uid:05d}", age, gender, arche])

users_full = pd.DataFrame(user_rows, columns=["UserID", "Age", "Gender", "_Archetype"])
users = users_full[["UserID", "Age", "Gender"]].copy()  # public sheet (no archetype)

# ---------------------------------------------------------------
# 3. TRANSACTIONS  (driven by archetype)
# ---------------------------------------------------------------
date_start = pd.Timestamp("2024-01-01")
date_end = pd.Timestamp("2026-06-01")
date_range_days = (date_end - date_start).days

cat_list = categories
courses_by_cat = {c: courses[courses.CourseCategory == c] for c in cat_list}

def sample_courses_for_user(arche, rng):
    """Return list of CourseIDs enrolled, with archetype-specific behavior."""
    if arche == "Explorer":
        n = rng.integers(6, 14)
        chosen_cats = rng.choice(cat_list, size=min(len(cat_list), rng.integers(4, 7)), replace=False)
        level_p = [0.55, 0.35, 0.10]
    elif arche == "Specialist":
        n = rng.integers(5, 11)
        primary = rng.choice(cat_list)
        chosen_cats = [primary] * 3 + list(rng.choice(cat_list, size=1, replace=False))
        level_p = [0.20, 0.45, 0.35]
    elif arche == "Career-Switcher":
        n = rng.integers(4, 9)
        target = rng.choice(["Data Science", "Web Development", "Cloud Computing", "Cybersecurity"])
        chosen_cats = [target] * 2 + ["Business"]
        level_p = [0.40, 0.35, 0.25]
    elif arche == "Casual Dabbler":
        n = rng.integers(1, 4)
        chosen_cats = rng.choice(cat_list, size=min(len(cat_list), rng.integers(1, 3)), replace=False)
        level_p = [0.75, 0.20, 0.05]
    else:  # Premium Upskiller
        n = rng.integers(3, 8)
        chosen_cats = rng.choice(cat_list, size=min(len(cat_list), rng.integers(2, 4)), replace=False)
        level_p = [0.10, 0.30, 0.60]

    picks = []
    for _ in range(n):
        cat = rng.choice(chosen_cats)
        pool = courses_by_cat[cat]
        if len(pool) == 0:
            continue
        lvl = rng.choice(levels, p=level_p)
        sub = pool[pool.CourseLevel == lvl]
        if len(sub) == 0:
            sub = pool
        row = sub.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        picks.append(row.CourseID)
    return list(dict.fromkeys(picks))  # de-dup, preserve order

txn_rows = []
tid = 1
for _, urow in users_full.iterrows():
    arche = urow["_Archetype"]
    course_ids = sample_courses_for_user(arche, rng)
    # enrollment frequency: premium upskillers & career switchers enroll in denser bursts
    spread = {"Explorer": 0.9, "Specialist": 0.7, "Career-Switcher": 0.5,
              "Casual Dabbler": 1.0, "Premium Upskiller": 0.4}[arche]
    anchor_day = rng.integers(0, date_range_days)
    for cidx, cid_ in enumerate(course_ids):
        offset = int(rng.normal(0, 60 * spread)) + cidx * int(rng.integers(5, 40))
        day = int(np.clip(anchor_day + offset, 0, date_range_days))
        tdate = date_start + pd.Timedelta(days=day)
        crow = courses.loc[courses.CourseID == cid_].iloc[0]
        price = crow.BasePrice
        # premium upskillers pay full/near-full price, casual dabblers get more discounts
        discount = {"Explorer": 0.85, "Specialist": 0.85, "Career-Switcher": 0.80,
                    "Casual Dabbler": 0.65, "Premium Upskiller": 0.95}[arche]
        amount = round(max(5, price * discount * rng.normal(1.0, 0.08)), 2)
        txn_rows.append([f"T{tid:06d}", urow.UserID, cid_, tdate.strftime("%Y-%m-%d"), amount])
        tid += 1

transactions = pd.DataFrame(txn_rows, columns=["TransactionID", "UserID", "CourseID", "TransactionDate", "Amount"])

# Save
courses_out = courses[["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating"]]
users.to_csv(os.path.join(_DATA_DIR, "users.csv"), index=False)
courses_out.to_csv(os.path.join(_DATA_DIR, "courses.csv"), index=False)
transactions.to_csv(os.path.join(_DATA_DIR, "transactions.csv"), index=False)
users_full.to_csv(os.path.join(_DATA_DIR, "_users_with_archetype_groundtruth.csv"), index=False)

print("Users:", users.shape)
print("Courses:", courses_out.shape)
print("Transactions:", transactions.shape)
print(transactions.head())
print("\nUsers with no transactions:", users.shape[0] - transactions.UserID.nunique())
