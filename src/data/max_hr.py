def max_hr(athletes_list):
    """
    This function calculates the maximum heart rate of athletes in a list.
    It does this by going through each athlete's entire metadata and identifying the all time maximum heart rate.

    Parameters:
    athletes (list): A list of dictionaries where each dictionary contains the metadata of an athlete.

    Returns:
    max_hr (int): The maximum heart rate of all athletes in the list.
    """
    # Importing necessary packages
    import pandas as pd
    from opendata import OpenData

    od = OpenData()

    # Creating an empty dataframe to store the maximum heart rate of each athlete
    df_max_hr = pd.DataFrame()

    # Looping through each athlete in the accepted list
    for athleteId in athletes_list:
        athlete = od.get_remote_athlete(athleteId)  # Accesses the athlete's data
        activities = athlete.activities()  # Accesses the athlete's activities

        max_hr = 0  # Initializing the maximum heart rate

        # Looping through each activity to find the maximum heart rate
    for activity in activities:
        if activity.metadata["sport"] == "Bike":
            if (
                "METRICS" in activity.metadata
                and "max_heartrate" in activity.metadata["METRICS"]
            ):
                ride_max_hr = float(activity.metadata["METRICS"]["max_heartrate"])
                if ride_max_hr > max_hr:
                    max_hr = ride_max_hr

        # Appending the maximum heart rate of the athlete to the dataframe
        df_max_hr["athleteId"] = athleteId
        df_max_hr["max_hr"] = max_hr

        # Returning the dataframe
        return df_max_hr
