# Uber Ride Analysis Project

Name: Jayakumar P
Dataset: uberdrive.csv (My Uber Drives dataset)

## About

This project analyzes personal Uber ride data from 2016. The dataset has
1156 rides with columns for start/end time, category (Business/Personal),
start/stop location, miles, and purpose of the trip.

Note: this dataset does not have a fare/price column, only miles and
duration. So for the "fare trends" part I used miles as a proxy since
Uber pricing depends mostly on distance and time.

## Files

- `data/uberdrive.csv` - the raw dataset
- `uber_analysis.py` - the main code (cleaning + EDA + charts + model)
- `uber_clean.csv` - cleaned data (gets created after running the script)
- charts (.png files) - also get created after running the script

## What the code does

1. Loads the csv and checks the data
2. Cleans it up:
   - removes the `*` from column names
   - removes the "Totals" row at the bottom
   - fixes date columns (this dataset has 2 different date formats mixed
     together which caused a lot of errors at first, had to use
     `format="mixed"` in pd.to_datetime to fix it)
   - drops rows with missing/invalid miles or duration
   - fills missing purpose values with "Not Specified"
3. Prints basic stats (mean miles, category counts, purpose counts etc)
4. Makes charts:
   - rides by hour of day
   - rides by day of week
   - heatmap of day vs hour (peak hours)
   - monthly trend
   - purpose breakdown
   - business vs personal pie chart
   - distance distribution
   - top 10 pickup locations
5. Builds a simple Random Forest model to predict ride demand (number of
   rides) based on hour, day of week and month

## Key Insights

- Total rides after cleaning: 1151
- Total miles: ~12,130
- Average trip: ~10.5 miles
- Most rides are Business category (1074 vs 77 Personal)
- Busiest hour: around 3 PM
- Busiest day: Friday
- Busiest month: December
- Most common purpose (that's actually filled in): Meeting

## Model results

Used a Random Forest Regressor to predict ride count using hour, day of
week, and month as features.

- MAE: ~0.25
- R2 Score: came out low (even negative on some runs)

This is because the dataset is just one person's ride history, not a
big city-wide dataset, so most hour/date combinations only have 0 or 1
rides. Not really enough repeating pattern for the model to learn well.
Still, the feature importance shows HOUR and MONTH matter the most,
which matches what we already saw in the EDA charts.

## Libraries used

pandas, numpy, matplotlib, seaborn, scikit-learn

## How to run

```
pip install pandas numpy matplotlib seaborn scikit-learn
python uber_analysis.py
```

## Possible improvements

- Real fare data would let us predict actual price instead of just miles
- A bigger dataset (multiple users) would make the demand model more
  accurate and useful for things like driver allocation or surge pricing
