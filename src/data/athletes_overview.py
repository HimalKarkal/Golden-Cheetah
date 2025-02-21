"""
This script collects information about each athlete in the Golden Cheetah dataset and creates a CSV file with the following columns:

- id: The athlete's unique identifier
- gender: The athlete's gender
- yob: The athlete's year of birth
- numberOfRides: The total number of bike rides recorded for the athlete
- duration: The number of days between the first and last bike ride recorded for the athlete
- rideFrequency: The average number of bike rides per day for the athlete

This CSV is saved in the data\processed directory as athlete_survey.csv and is used to select athletes for further analysis in the project.
"""

# Importing packages
from opendata import OpenData
import polars as pl
from datetime import datetime

od = OpenData()

# Getting a directory of all athletes in the dataset
athletes = od.remote_athletes()

# List to hold data for each athlete
athlete_records = []

for athlete in athletes:
    try:
        metadata = athlete.metadata
        if metadata is not None:
            record = {}
            # Collect athlete ID, gender, and year of birth
            record["id"] = metadata["ATHLETE"]["id"][1:-1]  # Remove curly braces
            record["gender"] = metadata["ATHLETE"]["gender"]
            record["yob"] = metadata["ATHLETE"]["yob"]

            # Isolate bike rides from all activities
            dict_BikeRides = {}
            for key in metadata["RIDES"].keys():
                if metadata["RIDES"][key]["sport"] == "Bike":
                    dict_BikeRides[key] = metadata["RIDES"][key]

            # Total number of rides
            record["numberOfRides"] = len(dict_BikeRides)

            # Calculate duration between first and last bike ride
            keys_list = list(dict_BikeRides.keys())
            if len(keys_list) != 0:
                startDate_str = keys_list[0].split(" ")[0]
                endDate_str = keys_list[-1].split(" ")[0]
                try:
                    startDate = datetime.strptime(startDate_str, "%Y/%m/%d")
                    endDate = datetime.strptime(endDate_str, "%Y/%m/%d")
                    record["duration"] = (endDate - startDate).days
                except ValueError:
                    record["duration"] = 0
            else:
                record["duration"] = 0

            # Calculate ride frequency (rides per day)
            record["rideFrequency"] = (
                (record["numberOfRides"] / record["duration"])
                if record["duration"] != 0
                else 0
            )

            athlete_records.append(record)
            print(f"{len(athlete_records)} Processed athlete {athlete}")

    except Exception as e:
        print(f"Skipping athlete {athlete} due to error: {e}")
        continue

# Create the final Polars DataFrame in one go
df_AthleteSurvey = pl.DataFrame(athlete_records)

# Save the DataFrame to a CSV file
df_AthleteSurvey.write_csv(r"..\..\data\processed\athletes_overview.csv")
