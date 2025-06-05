# Imports
import polars as pl
import numpy as np
from opendata import OpenData
from botocore.exceptions import ClientError
from rust_utils import hampel_filter


# Importing ActivityFunctions class
from .activity_funcs import ActivityFunctions

# Creating an OpenData object
od = OpenData()


class Athlete:
    def __init__(self, athlete_id):
        self.id = athlete_id

        # Try getting athlete data locally
        try:
            self.activities = list(
                od.get_local_athlete(athlete_id=self.id).activities()
            )
            self.metadata = od.get_local_athlete(athlete_id=self.id).metadata
            print("Athlete data loaded successfully from local storage.")

        # If athlete data not found locally, fetch from remote storage and store locally before loading.
        except FileNotFoundError:
            print(
                "Athlete data not found in local storage. Attempting to load from remote storage."
            )
            try:
                od.get_remote_athlete(athlete_id=self.id).store_locally()
                self.activities = list(
                    od.get_local_athlete(athlete_id=self.id).activities()
                )
                self.metadata = od.get_local_athlete(athlete_id=self.id).metadata
                print("Athlete data loaded successfully from remote storage.")

            # If the athlete ID is invalid, ask the user to check the athlete ID.
            except ClientError as ex:
                if ex.response["Error"]["Code"] == "NoSuchKey":
                    print("Athlete not found! Provide a valid athlete ID.")

    def get_hr_min_max(self):
        """Identifies the minimum and maximum heart rate (HR) for the athlete after filtering outliers from HR series data."""
        # Creating an empty list to store lists of hr series
        max_hr_array = np.array([])
        min_hr_array = np.array([])

        # Iterating through each activity
        for activity in self.activities:
            # Checking whether the activity is a bike ride
            if activity.metadata["sport"] == "Bike":
                # Checking whether the activity has hr data
                # Filtering outliers from the HR series using the Hampel filter
                hr_series = activity.data["hr"]

                # Continue if the heart rate data is too short or contains only missing values
                if len(hr_series) < 10 or hr_series.isna().all():
                    continue

                # Applying the Hampel filter to the heart rate data
                hr_series = hampel_filter(
                    hr_series.to_list(), half_window=10, n_sigma=3.0
                )
                # Appending the maximum heart rate from the filtered HR series to max_hr_array
                max_hr_array = np.append(max_hr_array, max(hr_series))

                # Appending the minimum heart rate from the filtered HR series to min_hr_array
                min_hr_array = np.append(min_hr_array, min(hr_series))

        # Getting overall max HR from max_hr_array
        overall_max_hr = max_hr_array.max()

        # Removing zeroes from the min_hr_array
        min_hr_array = min_hr_array[min_hr_array != 0]

        # Getting overall min HR from min_hr_array
        overall_min_hr = min_hr_array.min()

        self.max_hr = overall_max_hr
        self.min_hr = overall_min_hr
        print(f"Minimum heart rate for athlete {self.id} is {self.min_hr} bpm.")
        print(f"Maximum heart rate for athlete {self.id} is {self.max_hr} bpm.")

    def identify_min_hr(self):
        pass

    def process_hrr(self, max_hr):
        # Creating an empty list to store the processed dataframes
        processed_dfs = []

        # Iterating through each activity
        for activity in self.activities:
            # Applying the ActivityFunctions.process_hrr method to each activity
            df = ActivityFunctions.process_hrr(
                activity_instance=activity, max_hr=max_hr
            )

            # Adding the processed dataframe to the list
            if not df.is_empty():
                processed_dfs.append(df)

        # Concatenating all processed dataframes into a single dataframe
        processed_df = pl.concat(processed_dfs)

        # Adding athlete ID column to the processed dataframe
        processed_df = processed_df.with_columns(pl.lit(self.id).alias("athleteID"))
        processed_df = processed_df.select(
            ["athleteID", "date", "startTime", "endTime", "HRR(30)"]
        )

        # Returning the processed dataframe
        return processed_df

    def process_critical_power(self):
        pass

    def process_trimp(self):
        pass
