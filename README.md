# Golden Cheetah
  
## src\data
This directory contains python scripts that are used for data processing. It contains the following files:
1. athletes_overview.py
2. heart_rate.py
3. activity_funcs.py
4. athlete_funcs.py

The purpose of each is described below.

### 1. athletes_overview.py
This python script creates a CSV file containing information about each athlete in the Golden Cheetah open dataset. It accesses the Amazon Web Services S3 bucket through the OpenData package developed by the Golden Cheetah team. It then loops through each athlete and collects the following information:

1. **id**: The athlete's unique identifier
2. **sex**: The athlete's sex
3. **yob**: The athlete's year of birth
4. **numberOfRides**: The total number of bike rides recorded for the athlete
5. **duration**: The number of days between the first and last bike ride recorded for the athlete
6. **rideFrequency**: The average number of bike rides per day for the athlete
  
The resulting dataframe is saved to csv format and named *athletes_overview*. As of 7/1/2025, this file contained 6576 rows of data.

### 2. heart_rate.py
This python script defines a function to calculate the maximum heart rate for any athlete. It accepts an athlete id, retrives the athlete's data from the Golden Cheetah AWS bucket, and loops through all their ride activity files. For each ride, it applies the hampel filter imported from sktime to remove outliers, then determines the maximum heart rate during the ride. This is then compared to the overall heart rate of the athlete. The athlete's overall maximum heart rate is updated if maximum heart rate for any ride is greater than the overall maximum heart rate.

The hr_max() function defined in this script was called in *notebooks\0.02_heart_rate_filtering.ipynb* to create a final list of athletes included in the study.

This script is very inefficient. A faster implementation of the hampel filter written in rust is used in the *src\data\athlete_class.py* file to identify both the maximum and the minimum heart rate for an athlete. 

### 3. athlete_class.py
This python script defines the ActivityFunctions class that contains static methods to perform three operations:
    1. **process_hrr**: This method accepts an activity and the athlete's maximum heart rate and calculates HRR(30) instances from the data. Windows where a HRR(30) calculation can be made must meet the following conditions:
        i. Window is continuous and at least 30 seconds long,
        ii. Power output remains at or under 20 watts throughout the window,
        iii. The heart rate at the start of the window must be at greater than or equal to 80% of the athlete's maximum heart rate.
    For each window, the highest HRR is calculated and stored in a polars dataframe. A polars dataframe containing all HRR(30) readings from the current activity along with the start and end times of each window will be returned.

    2. **process_trimp**: In progress

    3. **process_cp**: In progress

It also defines an Athlete class that contains methods to calculate maximum and minimum heart rate, hrr, trimp, and cp for all activities of an athlete. It loops through each activity and applies the relevant ActivityFunctions method.



