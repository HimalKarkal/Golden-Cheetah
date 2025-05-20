# Imports
import polars as pl
from opendata import OpenData
from botocore.exceptions import ClientError

# Importing ActivityFunctions class
from activity_funcs import ActivityFunctions

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

    def identify_max_hr(self):
        pass

    def identify_min_hr(self):
        pass

    def process_hrr(self, max_hr):
        # Creating an empty list to store the processed dataframes
        processed_dfs = []

        # Iterating through each activity
        for activity_instance in self.activities:
            # Applying the ActivityFunctions.process_hrr method to each activity
            df = ActivityFunctions.process_hrr(
                activity_instance=activity_instance, max_hr=max_hr
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
