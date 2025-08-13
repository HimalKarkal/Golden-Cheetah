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
    def process_hrr(
        activity_instance: models.Activity, max_hr: int
    ) -> pl.DataFrame | None:
        """Processes activity data and returns a dataframe on heart rate recovery.
        Args:
            activity_instance: An instance of opendata.models.Activity.
            max_hr: The maximum heart rate for the athlete.

        Returns:
            An output polars dataframe containing the highest HRR collected from the activity.
        """

        # Loading data into polars dataframe
        if activity_instance.data is None:
            return None
        elif activity_instance.metadata is None:
            return None
        elif activity_instance.data["hr"].isna().all():
            return None
        else:
            # Converting the activity data to a polars dataframe
            df = pl.from_pandas(activity_instance.data)

        # Filtering only rows whose power output is less than 20 watts and excluding rows with HR values less than 25 bpm
        df = df.filter(pl.col("power") <= 20, pl.col("hr") >= 25)

        # Identifying continuous sequences in the dataframe by seconds.
        df = df.with_columns(
            pl.col("secs").diff().fill_null(1).ne(1).cum_sum().alias("sequence_number")
        )  # Calculates row-by-row difference in seconds data, then assigns sequence numbers.

        # Filtering those sequence numbers that are at least 30 rows long.
        df = df.filter(
            pl.len().over("sequence_number") >= 30,
        )

        # There is a chance that the dataframe is empty at this point.
        # All sequences could have been less than 30 rows long.
        # We will return None if that is the case

        if df.is_empty():
            return None

        # Applying the hampel filter to the hr column for each continuous segment
        filtered_segment_dfs = []
        for sequence_no in df["sequence_number"].unique():
            df_temp = df.filter(pl.col("sequence_number") == sequence_no)
            hr_filtered = hampel_filter(
                df_temp["hr"].to_list(), half_window=10, n_sigma=3.0
            )
            hr_filtered = [None] * 10 + hr_filtered[10:-10] + [None] * 10
            df_temp = df_temp.with_columns(pl.Series(name="hr", values=hr_filtered))
            df_temp = df_temp.drop_nulls()
            df_temp = df_temp.with_columns(
                pl.col("hr").diff().fill_null(0).alias("hr_delta")
            )
            filtered_segment_dfs.append(df_temp)

        df = pl.concat(filtered_segment_dfs)

        # Calculating HR decrease over 30 seconds in all remaining sequences
        df = df.with_columns(
            (-pl.col("hr_delta"))  # Convert HR drops to positive values
            .rolling_sum(
                window_size=30, min_samples=30
            )  # Sum these drops over a 30-sample window
            .over("sequence_number")  # Perform this calculation within each sequence
            .alias("hr_drop_in_30s_window"),
            pl.col(
                "hr"
            )  # Adding the athlete's hr at the start of the window to each row
            .shift(29)
            .alias("hr_at_window_start"),
        ).filter(
            # Keep only rows where a valid sum was computed and represents an actual drop (>=0)
            pl.col("hr_drop_in_30s_window").is_not_null()
            & (pl.col("hr_drop_in_30s_window") >= 0)
        )

        # Ensuring the athlete's HR at the start of each window is over 80% of their maximum
        df = df.filter(pl.col("hr_at_window_start") >= 0.8 * max_hr)

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

        # Creating output dataframe. Casting columns explicitly to prevent issues with concatenation.
        df = df.select(
            (pl.col("secs") - 29)
            .cast(pl.Int64)
            .alias("hrr_window_start_secs"),  # Start of the 30-second window
            pl.col("secs")
            .cast(pl.Int64)
            .alias("hrr_window_end_secs"),  # End of the 30-second window
            pl.col("hr_drop_in_30s_window")
            .cast(pl.Int64)
            .alias("HRR(30)"),  # The calculated max HRR for that window
        )

        # Adding columns for date and activity to output dataframe
        date = dt.datetime.strptime(
            activity_instance.metadata["date"], r"%Y/%m/%d %H:%M:%S UTC"
        )
        df = df.with_columns(
            pl.lit(date).alias("date"),
            pl.lit(activity_instance.id).alias("activity_id"),
        ).select(
            [
                "activity_id",
                "date",
                "hrr_window_start_secs",
                "hrr_window_end_secs",
                "HRR(30)",
            ]
        )

        return df

    @staticmethod
    def process_MaxMeanPower(
        activity_instance: models.Activity,
        max_hr: int,
        hr_threshold: float,
        window_len: int,
    ):
        """Processes activity and returns a dataframe on maximum mean power over the specified window size.
        Args:
            activity_instance (models.Activity): An instance of opendata.models.Activity.
            max_hr (int): The athlete's maximum heart rate.
            hr_threshold (float): The threshold percentage, in decimal format, for the athlete's mean heart rate over the window size for it to be considered a near maximal effort.
            window_len (int): The window size in minutes over which to calculate maximal mean power.
        Returns:
            pl.DataFrame: A polars dataframe containing the maximum mean power for each activity.
        """
        # Returning None if metadata does not exist
        if activity_instance.metadata is None:
            return None

        # Storing the date of the activity
        date = dt.datetime.strptime(
            activity_instance.metadata["date"],
            r"%Y/%m/%d %H:%M:%S UTC",
        )

        # Filtering outliers from the heart rate series using the Hampel filter
        activity_instance.data["hr"] = hampel_filter(
            activity_instance.data["hr"].to_list(), half_window=10, n_sigma=3.0
        )

        # Converting the activity data to a polars dataframe
        df = pl.from_pandas(activity_instance.data)

        # Identifying continuous segments in the dataframe and filtering out those that are too short
        df = df.with_columns(
            pl.col("secs").diff().ne(1).cum_sum().alias("segment_id")
        ).filter(pl.len().over("segment_id") >= window_len * 60)

        # Calculating the rolling average for power over POWER_WINDOW_LENGTH for each segment
        df = df.with_columns(
            pl.col("power")
            .rolling_mean(window_len * 60)
            .over("segment_id")
            .alias("rolling_mean_power"),
            pl.col("hr")
            .rolling_mean(window_len * 60)
            .over("segment_id")
            .alias("rolling_mean_hr"),
        )

        # Rolling mean power leaves null values at the start of each segment. We will drop these rows.
        # We will also drop segments where the athlete's rolling mean heart rate is below the threshold.
        # Finally, we will sort the dataframe in descending order of rolling mean power within each segment and select the first row of each sorted segment.
        df = (
            df.filter(
                pl.col("rolling_mean_power").is_not_null(),
                pl.col("rolling_mean_hr") >= max_hr * hr_threshold,
            )
            .sort(["segment_id", "rolling_mean_power"], descending=[False, True])
            .group_by("segment_id")
            .first()
        )

        # Adding athlete ID and date columns to the processed dataframe. Casting explicitly to prevent issues with concatenation.
        df_activity = df.with_columns(
            pl.lit(activity_instance.id).cast(pl.String).alias("activity_id"),
            pl.lit(date).cast(pl.Datetime).alias("date"),
            (pl.col("secs") - window_len * 60)
            .cast(pl.Int64)
            .alias("mmp_window_start_secs"),
            pl.col("secs").cast(pl.Int64).alias("mmp_window_end_secs"),
            pl.col("rolling_mean_power").cast(pl.Float64).alias("maximal_mean_power"),
        )

        # Removing unnecessary columns
        df_activity = df_activity.select(
            [
                "activity_id",
                "date",
                "mmp_window_start_secs",
                "mmp_window_end_secs",
                "maximal_mean_power",
            ]
        )

        return df_activity

    @staticmethod
    def process_trimp(
        activity_instance: models.Activity, gender: str, hr_max: int, hr_min: int
    ):
        """Processes activity and returns the TRIMP score for the activity"""

        # Check that the data is good
        if activity_instance.data["hr"] is None:
            return None
        elif activity_instance.metadata is None:
            return None
        elif activity_instance.data["hr"].isna().all():
            return None
        elif (activity_instance.data["hr"] == 0).all():
            return None
        else:
            df = pl.from_pandas(activity_instance.data)

        # Storing the date of the activity
        date = dt.datetime.strptime(
            activity_instance.metadata["date"],
            r"%Y/%m/%d %H:%M:%S UTC",
        )

        # Filtering out rows where HR is 0 bpm
        df = df.filter(pl.col("hr") >= 25)

        # Calculating average HR over activity
        hr_mean = df["hr"].mean()

        # Returning None if hr_mean is None
        if hr_mean is None:
            return None

        # Duration of activity (mins)
        duration = df.height / 60

        # Matching athlete to correct weighting factor based on gender
        match gender:
            case "M":
                Y = 1.92
            case "F":
                Y = 1.67

        # Calculating delta HR ratio
        delta_HR_ratio = (hr_mean - hr_min) / (hr_max - hr_min)

        # Banister's TRIMP calculation
        trimp = duration * delta_HR_ratio * Y

        output_df = pl.DataFrame(
            {"activity_id": activity_instance.id, "date": date, "trimp": trimp}
        )

        return output_df


# ATHLETE FUNCTIONS

# Creating an OpenData object
od = OpenData()


class Athlete:
    def __init__(self, athlete_id):
        self.id = athlete_id
        self.gender = None
        self.max_hr = None
        self.min_hr = None
        self.date_of_first_ride = None

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

    def get_gender(self):
        self.gender = self.metadata["ATHLETE"]["gender"]

    def get_date_of_first_ride(self):
        # Identifying the start date of each athlete's data
        # Creating an empty list to store dates
        dates = []
        # Iterating through each activity
        for activity in self.activities:
            if activity.metadata is None:
                continue
            if activity.metadata["sport"] == "Bike":
                dates.append(
                    dt.datetime.strptime(
                        activity.metadata["date"], r"%Y/%m/%d %H:%M:%S UTC"
                    )
                )

        # If no bike rides found, print a message and return None
        if dates == []:
            print(f"No bike rides found for athlete {self.id}.")
            self.date_of_first_ride = None
        else:
            # Getting the earliest date from the list of dates
            self.date_of_first_ride = min(dates)

    def get_hr_min_max(self):
        """Identifies the minimum and maximum heart rate (HR) for the athlete after filtering outliers from HR series data."""
        # Creating an empty list to store lists of hr series
        max_hr_array = np.array([])
        min_hr_array = np.array([])

        # Getting athlete gender
        if self.gender is None:
            self.get_gender()

        # Matching gender to hr cutoff
        match self.gender:
            case "M":
                cutoff = 215
            case "F":
                cutoff = 210

        # Iterating through each activity
        for activity in self.activities:
            # Checking whether the activity is a bike ride
            # Continue if activity has no metadata
            if activity.metadata is None:
                continue
            elif activity.data is None:
                continue
            else:
                hr_series = activity.data["hr"]

                # Removing values deemed impalusible from the hr series
                # Values less than 40 are highly unlikely considering the individuals are about to start exercising.
                # Similarly, values greater than 215 for males and 210 for females are unlikely. Supported by literature.
                hr_series = hr_series[
                    (hr_series >= 40) & (hr_series <= cutoff) & (~hr_series.isna())
                ]

                # Moving to the next iteration if hr_series is empty
                if len(hr_series) == 0:
                    continue

                # Appending the maximum heart rate from the filtered HR series to max_hr_array
                max_hr_array = np.append(max_hr_array, max(hr_series))

                # Appending the minimum heart rate from the filtered HR series to min_hr_array
                min_hr_array = np.append(min_hr_array, min(hr_series))

        # Calculating HR min and max as the 5th and 95th percentiles of their respective arrays if the array contains at least 20 elements
        no_of_readings = len(max_hr_array)
        print(no_of_readings == len(min_hr_array))
        if no_of_readings < 20:
            print(
                f"Athlete {self.id} does not have enough readings to identify max and min hr"
            )
        else:
            # Getting overall min and max HR from the sorted arrays as the 5th and 95th percentile respectively
            overall_min_hr = np.percentile(min_hr_array, 5)
            overall_max_hr = np.percentile(max_hr_array, 95)

            # Updating athlete data
            self.max_hr = overall_max_hr
            self.min_hr = overall_min_hr
            print(f"Minimum heart rate for athlete {self.id} is {self.min_hr} bpm.")
            print(f"Maximum heart rate for athlete {self.id} is {self.max_hr} bpm.")

    def process_hrr(self):
        """Processes HRR(30) data for the athlete across all bike rides and returns a polars dataframe with the results."""
        # Creating an empty list to store the processed dataframes
        processed_dfs = []

        # Checking if max_hr is set, if not, calling get_hr_min_max method to set it
        if self.max_hr is None:
            print("Athlete's max HR is unknown. Calculating it now.")
            self.get_hr_min_max()
        if self.date_of_first_ride is None:
            self.get_date_of_first_ride()

        # Iterating through each activity
        for activity in self.activities:
            # Checking whether the activity is a bike ride
            # Continue if activity has no metadata
            if activity.metadata is None:
                continue
            if activity.metadata["sport"] != "Bike":
                continue
            # Applying the ActivityFunctions.process_hrr method to each activity
            df = ActivityFunctions.process_hrr(
                activity_instance=activity, max_hr=self.max_hr
            )

            # Adding the processed dataframe to the list
            processed_dfs.append(df)

        # Dropping all None values from the processed_dfs list
        processed_df = [df for df in processed_dfs if df is not None]

        # If processed_df is empty, return an empty polars dataframe with the same schema as the output below
        if processed_df == []:
            return pl.DataFrame(
                {},
                schema=[
                    ("athlete_id", pl.String),
                    ("gender", pl.String),
                    ("week_no", pl.Int64),
                    ("activity_id", pl.String),
                    ("date", pl.Datetime),
                    ("hrr_window_start_secs", pl.Int64),
                    ("hrr_window_end_secs", pl.Int64),
                    ("HRR(30)", pl.Int64),
                ],
            )

        # Concatenating all processed dataframes into a single dataframe
        processed_df = pl.concat(processed_df)
        # Adding athlete ID column to the processed dataframe
        processed_df = processed_df.with_columns(
            pl.lit(self.id).alias("athlete_id"),
            (
                (pl.col("date") - self.date_of_first_ride).dt.total_days() // 7
            )  # Adding week number column
            .cast(pl.Int64)
            .alias("week_no"),
            pl.lit(self.metadata["ATHLETE"]["gender"]).alias("gender"),
        )

        processed_df = processed_df.select(
            [
                "athlete_id",
                "gender",
                "week_no",
                "activity_id",
                "date",
                "hrr_window_start_secs",
                "hrr_window_end_secs",
                "HRR(30)",
            ]
        )

        # Returning the processed dataframe
        return processed_df

    def process_mmp(self, hr_threshold: float, window_len: int):
        """Processed maximal mean power over the specified window size for all the athlete's bike rides and returns a polars dataframe"""
        # Creating an empty list to store the processed dataframes
        processed_dfs_list = []

        # Checking whether max_hr is set, if not, calling get_hr_min_max method to set it
        if self.max_hr is None:
            print("Athlete's max HR is unknown. Calculating it now.")
            self.get_hr_min_max()
        if self.date_of_first_ride is None:
            self.get_date_of_first_ride()

        # Iterating through each activity
        for activity in self.activities:
            # Skipping current iteration if activity is not a bike ride or the activity has no heart rate data
            # Continue if activity has no metadata
            if activity.metadata is None:
                continue
            if (
                activity.metadata["sport"] != "Bike"
                or activity.data["hr"].isna().all()
                or activity.data["power"].isna().all()
            ):
                continue

            # Applying the ActivityFunctions.process_MaxMeanPower method to each activity and appending the result to the list

            df_result = ActivityFunctions.process_MaxMeanPower(
                activity_instance=activity,
                max_hr=self.max_hr,
                hr_threshold=hr_threshold,
                window_len=window_len,
            )

            processed_dfs_list.append(df_result)

        if processed_dfs_list != []:
            # Concatenating all processed dataframes into a single dataframe
            output_df = (
                pl.concat(processed_dfs_list)
                .with_columns(
                    pl.lit(self.id).alias("athlete_id"),
                    pl.lit(self.metadata["ATHLETE"]["gender"]).alias("gender"),
                    ((pl.col("date") - self.date_of_first_ride).dt.total_days() // 7)
                    .cast(pl.Int64)
                    .alias("week_no"),
                )
                .select(
                    [
                        "athlete_id",
                        "gender",
                        "week_no",
                        "activity_id",
                        "date",
                        "mmp_window_start_secs",
                        "mmp_window_end_secs",
                        "maximal_mean_power",
                    ]
                )
            )
            return output_df
        else:
            return pl.DataFrame(
                {},
                schema=[
                    ("athlete_id", pl.String),
                    ("gender", pl.String),
                    ("week_no", pl.Int64),
                    ("activity_id", pl.String),
                    ("date", pl.Datetime),
                    ("mmp_window_start_secs", pl.Int64),
                    ("mmp_window_end_secs", pl.Int64),
                    ("maximal_mean_power", pl.Float64),
                ],
            )

    def process_trimp(self):
        # Checking whether the athlete's gender, max_hr, min_hr, week number, and date of first ride are loaded
        if self.max_hr is None:
            self.get_hr_min_max()
        if self.date_of_first_ride is None:
            self.get_date_of_first_ride()
        if self.gender is None:
            self.get_gender()

        # Looping through activities to identify

        processed_dfs_list = [
            ActivityFunctions.process_trimp(
                activity, self.gender, self.max_hr, self.min_hr
            )
            for activity in self.activities
        ]
        processed_dfs_list = [
            df for df in processed_dfs_list if df is not None
        ]  # Removing None from the list

        # Creating output dataframe
        output_df = pl.concat(processed_dfs_list)

        # Adding athlete_id, gender, and week_no to output_df
        output_df = output_df.with_columns(
            pl.lit(self.id).cast(pl.String).alias("athlete_id"),
            pl.lit(self.gender).cast(pl.String).alias("gender"),
            ((pl.col("date") - self.date_of_first_ride).dt.total_days() // 7).alias(
                "week_no"
            ),
        ).select(["athlete_id", "gender", "week_no", "activity_id", "date", "trimp"])

        output_df = (
            output_df.group_by(["athlete_id", "gender", "week_no"])
            .agg(pl.sum("trimp").alias("total_weekly_trimp"))
            .sort("week_no")
        )

        return output_df
