# Golden Cheetah Data Processing

This repository contains Python scripts used for processing data from the Golden Cheetah open dataset. The scripts access athlete and activity data from an Amazon Web Services (AWS) S3 bucket and perform various data cleaning, transformation, and analysis tasks.

---

## `src\data`

This directory contains the core Python scripts for all data processing tasks.

### 1. `athletes_overview.py`

This script creates a CSV file named `athletes_overview.csv` containing summary information for each athlete in the Golden Cheetah open dataset. It accesses the AWS S3 bucket via the `OpenData` package provided by the Golden Cheetah team.

The script iterates through each athlete and gathers the following details:

* **id**: The athlete's unique identifier
* **sex**: The athlete's sex
* **yob**: The athlete's year of birth
* **numberOfRides**: The total number of bike rides recorded for the athlete
* **duration**: The number of days between the first and last recorded bike ride
* **rideFrequency**: The average number of bike rides per day for the athlete

As of 7/1/2025, the generated file contained 6,576 rows of data.

### 2. `heart_rate.py`

This script defines a function to calculate the maximum heart rate for any given athlete. It retrieves an athlete's data from the Golden Cheetah AWS bucket and iterates through all their ride activity files.

For each ride, it applies the `hampel` filter (imported from `sktime`) to remove outliers from the heart rate data before determining the maximum heart rate for that ride. This value is then compared against the athlete's overall maximum heart rate, which is updated if a new maximum is found.

**Note:** The `hr_max()` function defined in this script was utilized in `notebooks\0.02_heart_rate_filtering.ipynb` to create the final list of athletes for the study. This script is highly inefficient. A faster implementation of the Hampel filter, written in Rust, is used in the `src\data\athlete_class.py` file to identify both the maximum and minimum heart rate for an athlete.

### 3. `athlete_class.py`

This script defines two primary components for advanced data processing: the `ActivityFunctions` class and the `Athlete` class.

#### `ActivityFunctions` Class

This class contains static methods to perform several key operations on activity data:

1.  **`process_hrr`**: This method calculates Heart Rate Recovery (HRR(30)) instances from a given activity and the athlete's maximum heart rate. Windows for HRR(30) calculation must meet these conditions:
    * The window must be continuous and at least 30 seconds long.
    * Power output must remain at or below 20 watts throughout the window.
    * The heart rate at the start of the window must be at or above 80% of the athlete's maximum heart rate.

    The highest HRR from each valid window is stored in a Polars DataFrame, which is returned along with the start and end times of each window.

2.  **`process_trimp`**: *In progress*

3.  **`process_criticalPower`**: *In progress*

#### `Athlete` Class

This class provides methods to calculate metrics such as maximum and minimum heart rate, HRR, TRIMP, and Critical Power (CP) for all activities associated with an athlete. It iterates through each of an athlete's activities and applies the relevant methods from the `ActivityFunctions` class.