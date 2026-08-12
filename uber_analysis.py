
# Uber Ride Analysis Project
# Name: Jayakumar P
# Dataset: uberdrive.csv (My Uber Drives dataset from Kaggle)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------
# STEP 1: Load the data
# ------------------------------------------------
data = pd.read_csv("data/uberdrive.csv")

print(data.head())
print(data.shape)
print(data.info())

# ------------------------------------------------
# STEP 2: Clean the data
# ------------------------------------------------

# column names have a * at the end, remove it
data.columns = [col.replace("*", "") for col in data.columns]

# last row is just a "Totals" row, not a real trip, so remove it
data = data[data["START_DATE"] != "Totals"]

# the dates in this csv are in two different formats which was annoying,
# some rows are like 01-01-2016 21:11 and some are like 1/13/2016 13:54
# so normal pd.to_datetime() failed on half the rows. Had to use format="mixed"
data["START_DATE"] = pd.to_datetime(data["START_DATE"], format="mixed")
data["END_DATE"] = pd.to_datetime(data["END_DATE"], format="mixed")

# drop rows with missing miles
data = data.dropna(subset=["MILES"])

# fill missing purpose with "Not Specified" instead of dropping (too many blanks)
data["PURPOSE"] = data["PURPOSE"].fillna("Not Specified")
data["CATEGORY"] = data["CATEGORY"].fillna("Unknown")

# add duration column in minutes
data["DURATION_MIN"] = (data["END_DATE"] - data["START_DATE"]).dt.total_seconds() / 60

# remove weird rows where duration is 0 or negative (bad data)
data = data[data["DURATION_MIN"] > 0]
data = data[data["MILES"] > 0]

# add some new columns for analysis
data["HOUR"] = data["START_DATE"].dt.hour
data["DAY"] = data["START_DATE"].dt.day_name()
data["MONTH"] = data["START_DATE"].dt.month_name()

print("\nAfter cleaning:", data.shape)

data.to_csv("uber_clean.csv", index=False)

# ------------------------------------------------
# STEP 3: Basic stats
# ------------------------------------------------
print("\n--- Basic Stats ---")
print(data["MILES"].describe())
print(data["CATEGORY"].value_counts())
print(data["PURPOSE"].value_counts())

total_miles = data["MILES"].sum()
avg_miles = data["MILES"].mean()
print("Total miles:", total_miles)
print("Average miles per trip:", avg_miles)

# ------------------------------------------------
# STEP 4: Visualizations
# ------------------------------------------------
sns.set_style("whitegrid")

# rides by hour
plt.figure(figsize=(10, 5))
data["HOUR"].value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Rides by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig("rides_by_hour.png")
plt.show()

# rides by day of week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(9, 5))
data["DAY"].value_counts().reindex(day_order).plot(kind="bar", color="orange")
plt.title("Rides by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig("rides_by_day.png")
plt.show()

# heatmap of hour vs day (peak hours)
pivot_table = data.pivot_table(index="DAY", columns="HOUR", values="MILES", aggfunc="count", fill_value=0)
pivot_table = pivot_table.reindex(day_order)
plt.figure(figsize=(12, 5))
sns.heatmap(pivot_table, cmap="YlOrRd")
plt.title("Peak Hours Heatmap (Day vs Hour)")
plt.tight_layout()
plt.savefig("peak_hours_heatmap.png")
plt.show()

# monthly trend
month_order = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
plt.figure(figsize=(10, 5))
data["MONTH"].value_counts().reindex(month_order).plot(kind="line", marker="o", color="green")
plt.title("Monthly Ride Trend")
plt.xlabel("Month")
plt.ylabel("Number of Rides")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_trend.png")
plt.show()

# purpose breakdown
plt.figure(figsize=(9, 5))
data["PURPOSE"].value_counts().plot(kind="barh", color="purple")
plt.title("Rides by Purpose")
plt.xlabel("Number of Rides")
plt.tight_layout()
plt.savefig("purpose_breakdown.png")
plt.show()

# business vs personal pie chart
plt.figure(figsize=(6, 6))
data["CATEGORY"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Business vs Personal Rides")
plt.ylabel("")
plt.tight_layout()
plt.savefig("category_split.png")
plt.show()

# miles distribution (used as a substitute for fare, since dataset has no fare column)
plt.figure(figsize=(9, 5))
sns.histplot(data[data["MILES"] < 50]["MILES"], bins=30, kde=True, color="red")
plt.title("Distribution of Trip Distance (miles)")
plt.xlabel("Miles")
plt.tight_layout()
plt.savefig("miles_distribution.png")
plt.show()

# top pickup locations
plt.figure(figsize=(9, 5))
data["START"].value_counts().head(10).plot(kind="barh", color="teal")
plt.title("Top 10 Pickup Locations")
plt.xlabel("Number of Rides")
plt.tight_layout()
plt.savefig("top_locations.png")
plt.show()

# ------------------------------------------------
# STEP 5: Simple prediction model - ride demand forecasting
# ------------------------------------------------
# group data into date+hour buckets to get ride counts per hour
data["DATE"] = data["START_DATE"].dt.date
demand = data.groupby(["DATE", "HOUR"]).size().reset_index(name="RIDE_COUNT")
demand["DATE"] = pd.to_datetime(demand["DATE"])
demand["DAY_NUM"] = demand["DATE"].dt.dayofweek
demand["MONTH_NUM"] = demand["DATE"].dt.month

X = demand[["HOUR", "DAY_NUM", "MONTH_NUM"]]
y = demand["RIDE_COUNT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n--- Model Results ---")
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(7, 4))
importance.plot(kind="bar", color="darkblue")
plt.title("Feature Importance for Demand Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nDone! All charts saved in the project folder.")
