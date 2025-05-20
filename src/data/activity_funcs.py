import opendata.models as models
import polars as pl
import datetime as dt


class ActivityFunctions:
    @staticmethod
    def max_hr(activity_instance: models.Activity):
        """Identifies maximum heart rate from the activity data.
        Args:
              activity_instance: An instance of opendata.models.Activity.

        Returns:
              The maximum heart rate from the activity data as an integer.
        """
        if activity_instance.data is None:
            return None
        else:
            df = activity_instance.data
        return df["hr"].max()

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

        # Adding columns for date, and athlete ID to output dataframe
        date = activity_instance.metadata["date"]
        date = dt.datetime.strptime(date, "%Y/%m/%d %H:%M:%S UTC")
        df = df.with_columns(pl.lit(date).alias("date"))
        df = df.select(["date", "startTime", "endTime", "HRR(30)"])

        return df

    def process_critical_power(self):
        pass

    def process_trimp(self):
        pass
