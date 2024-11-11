## Data Collection

To begin, we first import the necessary libraries, namely numpy and pandas for data analysis, we also will used seaborn and matplotlib for the visualization and the machine learning model will be created with the help of Tensorflow

We get the dataset from NYC Taxi & Limousine Commission, for our research we use High Volume For-Hire Vehicle Trip Record data for the year 2023.

We will start by loading the dataset for January 2023 first and use matplotlib to get a better understanding of the data.

## Preprocessing

### Data Cleaning

First we will drop all rows with null values from the data

We want to use data from uber so will filter it out by selecting Uber's license number 'HV0003' from 'hvfhs_license_num'

Then we remove any extra columns leaving us with only the columns for 'request_datetime', 'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time'

Then we remove the entries for locations 264 and 265 as they are codes for unknown or outside NYC

We remove entries with the 'trip_time' or 'trip_miles' which are more than the 99th percentile values

### Feature Engineering

Split the 'request_datetime' data into the hour of the day ('hour'), day of the week ('weekday') and week of the year ('week')

Convert the three features into cyclic features for better model learning using sin and cos

### Data Transformation

We use these features to group the data into demand for each Location at Each time value (15 minute intervals)

Using this we have demand for each location for all 15 minute intervals, we will use this data to split demand into categories of high, medium and low demand

(From processing the information it was found that having the bins as 0-8 for low 8-25 for medium and above 25 for high gave a somewhat even distribution for the count of the values)

Now we repeat the process for all other month data and append it to the new dataframe

Our final dataframe has the following columns

 0   Column         Dtype  
---  ------         -----  
 0   PULocationID   int64  
 1   hour           float64
 2   weekday        int32  
 3   week           UInt32 
 4   demand         int64  
 5   hour_sin       float64
 6   hour_cos       float64
 7   weekday_sin    float64
 8   weekday_cos    float64
 9   week_sin       Float64
 10  week_cos       Float64

## Model Training

### Training and Testing Data

The features we will use for prediciton are

hour_sin, hour_cos, weekday_sin, weekday_cos, week_sin, week_cos, PULocationID

The field we will be prediciting is

demand_category

We split the X and y values into train and test data using sklearn's train_test_split function

### Model Creation

We use a tensorflow Sequential model for the prediction

The first layer is a dense layer with 128 neurons

The second layer is a dense layer with 64 neurons

The third and output layer has 3 neurons for the three categories

We use a sparse_categorical_crossentropy loss function while compling the model

### Model Training

The model is fit using the training data with the test data being used for validation

