# Golden Cheetah
  
## athletes_overview.py
  
This python script creates a CSV file containing information about each athlete in the Golden Cheetah open dataset. It accesses the Amazon Web Services S3 bucket through the OpenData package developed by the Golden Cheetah team. It then loops through each athlete and collects the following information:

1. **id**: The athlete's unique identifier
2. **sex**: The athlete's sex
3. **yob**: The athlete's year of birth
4. **numberOfRides**: The total number of bike rides recorded for the athlete
5. **duration**: The number of days between the first and last bike ride recorded for the athlete
6. **rideFrequency**: The average number of bike rides per day for the athlete
  
The resulting dataframe is saved to  csv format and named *athletes_overview*. As of 7/1/2025, this file contained 6576 rows of data.

