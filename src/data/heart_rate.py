"""
This file contains helper functions to process heart rate data from the Golden Cheetah dataset.
The main function, hr_max(), calculates the maximum heart rate for a given athlete.

arguments: athlete_id (str) - the unique identifier for the athlete
returns: hr_max (int) - the maximum heart rate for the athlete
"""

import numpy as np
from opendata import OpenData
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.transformations.series.impute import Imputer

od = OpenData()

# Getting a directory of all athletes in the dataset
athletes = od.remote_athletes()


def hr_max(athlete_id: str):
    """
    This function calculates the maximum heart rate for a given athlete.

    arguments:
    athlete_id (str) - the unique identifier for the athlete

    returns:
    hr_max (int) - the maximum heart rate for the athlete
    """

    # Accessing the athlete's data
    athlete = od.get_remote_athlete(athlete_id)

    # Getting the athlete's activities
    activities = list(athlete.activities())

    # Initialising a variable to store the maximum heart rate
    overall_max_hr = 0

    # Looping through the athlete's activities
    for activity in activities:
        # Checking if the activity is a bike ride
        if activity.metadata["sport"] == "Bike":
            # Getting the heart rate data for the activity
            hr_series = activity.data["hr"]

            # Check whether hr_series contains nans
            if hr_series.isna().all():
                continue

            if hr_series.isna().any():
                hr_series = Imputer(method="linear").fit_transform(hr_series)

            # Applying the Hampel filter to the heart rate data
            hr_series_filtered = HampelFilter().fit_transform(hr_series)

            # Imputing missing values using the sktime imputer
            hr_series_imputed = Imputer(method="linear").fit_transform(
                hr_series_filtered
            )

            # Calculating the maximum heart rate for the activity
            activity_max_hr = np.max(hr_series_imputed)

            # Updating the overall maximum heart rate
            if activity_max_hr > overall_max_hr:
                overall_max_hr = activity_max_hr

    return overall_max_hr
