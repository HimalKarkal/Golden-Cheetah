"""
This script collects information about each athlete in the Golden Cheetah dataset and creates a csv file with the following columns:

- id: The athlete's unique identifier
- gender: The athlete's gender
- yob: The athlete's year of birth
- numberOfRides: The total number of bike rides recorded for the athlete
- duration: The number of days between the first and last bike ride recorded for the athlete
- rideFrequency: The average number of bike rides per day for the athlete

This csv is saved in the data\interim directory as athlete_survey.csv and is used to select athletes for further analysis in the project.
"""

# Importing packages
from opendata import OpenData
import pandas as pd
from datetime import datetime
import os

od = OpenData()

# Getting a directory of all athletes in the dataset
athletes = od.remote_athletes()

# Looping through each athlete and collecting information of interest

# Creating an empty pandas dataframe to store every athlete's information
df_AthleteSurvey = pd.DataFrame()

for athlete in athletes:
    try:  # Added a try-except block to handle errors for each athlete
        metadata = athlete.metadata

        if metadata is not None:
            df_athlete = pd.DataFrame()  # Creates a temporary empty dataframe to store an individual athlete's information

            # Collect athlete ID
            id = metadata["ATHLETE"]["id"][
                1:-1
            ]  # Collects id and slices curly braces out
            df_athlete["id"] = [id]  # Adds id to dataframe

            # Collect gender
            gender = metadata["ATHLETE"]["gender"]  # Collects gender
            df_athlete["gender"] = [gender]  # Adds gender to dataframe

            # Collect year of birth
            yearOfBirth = metadata["ATHLETE"]["yob"]  # Collects yearOfBirth
            df_athlete["yob"] = [yearOfBirth]  # Adds year_of_birth to dataframe

            # Isolate bike rides from all activities
            dict_BikeRides = {}  # Creates an empty dataframe to store only bike rides

            for key in metadata[
                "RIDES"
            ].keys():  # Loops through each activity in the RIDES dictionary
                if (
                    metadata["RIDES"][key]["sport"] == "Bike"
                ):  # Checks if the activity is a bike ride
                    dict_BikeRides[key] = metadata["RIDES"][
                        key
                    ]  # If True, adds the key-value pair to dict_BikeRides

            # Collect total number of rides
            numberOfRides = len(dict_BikeRides)
            df_athlete["numberOfRides"] = [
                numberOfRides
            ]  # Adds numberOfRides to dataframe

            # Collect duration of data collection
            keys_list = list(dict_BikeRides.keys())  # Converts dict_keys to a list
            if (
                len(keys_list) != 0
            ):  # Checks whether keys_list is empty, if not continues
                startDate = keys_list[0].split(" ")[
                    0
                ]  # Sets the first key's date as startDate
                endDate = keys_list[-1].split(" ")[
                    0
                ]  # Sets the last key's date as endDate
                try:  # Checks for Value Error. Observed one date recorded in a different language which threw an error
                    startDate = datetime.strptime(
                        startDate, "%Y/%m/%d"
                    )  # Converts startDate to datetime
                    endDate = datetime.strptime(
                        endDate, "%Y/%m/%d"
                    )  # Converts endDate to datetime
                    duration = (
                        endDate - startDate
                    ).days  # Calculates duration by subtracting the two datetime objects
                except ValueError:
                    pass
            else:  # If keys_list is empty sets duration to 0
                duration = 0

            df_athlete["duration"] = [duration]  # Adds duration to dataframe

            # Collect ride frequency
            rideFrequency = numberOfRides / duration if duration != 0 else 0
            df_athlete["rideFrequency"] = [
                rideFrequency
            ]  # Adds rideFrequency to dataframe

            # Add row to df_AthleteSurvey
            df_AthleteSurvey = pd.concat(
                [df_AthleteSurvey, df_athlete], ignore_index=True
            )

            # Print progress
            print(f"{len(df_AthleteSurvey)} Processed athlete {athlete}")

    except Exception as e:  # Catches any exception during processing of an athlete
        print(f"Skipping athlete {athlete} due to error: {e}")
        continue  # Continue with the next athlete

# Define the path to save the CSV file
output_dir = os.path.join("data", "interim")
output_file = os.path.join(output_dir, "athlete_survey.csv")

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the dataframe to a CSV file
df_AthleteSurvey.to_csv(output_file, index=False)
