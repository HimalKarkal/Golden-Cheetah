# Imports
import polars as pl
import numpy as np
import datetime as dt
from opendata import OpenData
import opendata.models as models
from botocore.exceptions import ClientError
from rust_utils import hampel_filter


# ACTIVITY FUNCTIONS
class ActivityFunctions:
    @staticmethod
    def process_hrr(activity_instance: models.Activity, max_hr: int):
        """Processes activity data and returns a dataframe on heart rate recovery.
        Args:
              activity_instance: An instance of opendata.models.Activity.

        Returns:
              An output dataframe containing information on HRR collected from the activity.
        """

        # Loading data into polars dataframe
        if activity_instance.data is None:
            return None
        else:
            df = pl.from_pandas(activity_instance.data)

        # Filtering only rows whose power output is less than 20 watts
        df = df.filter(pl.col("power") <= 20)

        # Calculating row-by-row difference in HR
        df = df.with_columns(pl.col("hr").diff().fill_null(0).alias("hr_delta"))

        # Identifying continuous sequences in the dataframe by seconds.
        df = df.with_columns(
            pl.col("secs").diff().fill_null(1).alias("secs_delta")
        )  # Calculates row-by-row difference in seconds data
        df = df.with_columns(
            pl.col("secs_delta").ne(1).cum_sum().alias("sequence_number")
        )  # Cumulative sum of secs_delta when a break is detected by [ne(1)]

        # Filtering those sequence numbers that are at least 30 rows long and whose first HR is over 80% max_hr
        df = df.filter(
            pl.len().over("sequence_number") >= 30
        )  # Ensures each sequence number is at least 30 rows long
        df = df.filter(
            pl.first("hr").over("sequence_number") >= 0.8 * max_hr
        )  # Ensures the first HR in each sequence is over 80% of HR max

        # Calculating HR decrease over 30 seconds in all remaining sequences
        df = df.with_columns(
            (-pl.col("hr_delta"))  # Convert HR drops to positive values
            .rolling_sum(
                window_size=30, min_samples=30
            )  # Sum these drops over a 30-sample window
            .over("sequence_number")  # Perform this calculation within each sequence
            .alias("hr_drop_in_30s_window")
        ).filter(
            # Keep only rows where a valid sum was computed and represents an actual drop (>=0)
            pl.col("hr_drop_in_30s_window").is_not_null()
            & (pl.col("hr_drop_in_30s_window") >= 0)
        )

        # Filtering out largest decreases in HR over a 30 second window
        df = (
            df.sort(
                by=["sequence_number", "hr_drop_in_30s_window"],
                descending=[
                    False,
                    True,
                ],  # Sort by sequence_id (asc), then by drop (desc)
            )
            .group_by(
                "sequence_number",
                maintain_order=True,  # Important because we sorted to get the max on top
            )
            .head(1)
        )  # Takes the first row for each group, which is the one with the max drop

        # Creating output dataframe
        df = df.select(
            (pl.col("secs") - 29).alias("startTime"),  # Start of the 30-second window
            pl.col("secs").alias("endTime"),  # End of the 30-second window
            pl.col("hr_drop_in_30s_window").alias(
                "HRR(30)"
            ),  # The calculated max HRR for that window
        )

        # Adding columns for date and activity to output dataframe
        date = activity_instance.metadata["date"]
        date = dt.datetime.strptime(date, "%Y/%m/%d %H:%M:%S UTC")
        df = df.with_columns(pl.lit(date).alias("date"))
        df = df.select(["date", "startTime", "endTime", "HRR(30)"])

        return df

    def process_critical_power(self):
        pass

    def process_trimp(self):
        pass


# ATHLETE FUNCTIONS

# Creating an OpenData object
od = OpenData()


class Athlete:
    def __init__(self, athlete_id):
        self.id = athlete_id
        self.max_hr = None
        self.min_hr = None

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

    def process_hrr(self):
        # Creating an empty list to store the processed dataframes
        processed_dfs = []

        # Checking if max_hr is set, if not, calling get_hr_min_max method to set it
        if self.max_hr is None:
            print("Athlete's max HR is unknown. Calculating it now.")
            self.get_hr_min_max()

        # Iterating through each activity
        for activity in self.activities:
            # Checking whether the activity is a bike ride
            if activity.metadata["sport"] != "Bike":
                continue
            # Applying the ActivityFunctions.process_hrr method to each activity
            df = ActivityFunctions.process_hrr(
                activity_instance=activity, max_hr=self.max_hr
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
