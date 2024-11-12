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

```
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
```

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

### Model Fitting

The model is fit using the training data with the test data being used for validation

## Model Testing

The model is tested first using tesnorflow model evaluation for metrics such as accuracy score

Then we make predicitons of the test data from the model and use scikit-learn metrics such as classification report and the confusion matrix

The confusion matrix recieved is

```
[[298051 153194 164989]
 [115057 209549 286249]
 [ 22716  95723 302969]]
```

We use seaborn and matplotlib to represent the confusion matrix in an easy to read format

## Real-Time Prediciton

### Prediciton
We want to predict the demand for all the different areas in the city at any specific point in time, to do so we will create a function that takes in a datetime value (defaults to current datetime value)

Using the datetime value we will first extract the hour, weekday and week values as in our training data, then we will convert them to cyclic values to align with the format that the model was trained on

Now we create a dataframe for all the LocationID values and add the time features to them, the dataframe should follow the format of our training data except it will have an entry for each LocationID for our selected time

Next we pass in this dataframe into the model for it to make predictions, the predicitons are saved along with their LocationIDs in a dataframe and returned out of the function

### Presentation
Now that we have demand categories for each loaction ID for our specified time we now need to visualize it

Using the taxi_zones data found at the NYC Taxi & Limousine Commission website we get the geographic polygonal data for each zone (Zones have been refered by PULoactionID or LocationID throughout our documentation) which can be plotted onto a map

This data is then joined with our output to get the demand for each zone or for the context of representation, the color

Using the folium and shapely libraries we initialize the map at NYC, then iterate through each entry in the dataset to add the respectively colored polygon onto the map

The map can be seen with green zones representing high demand areas, yellow zones representing medium demand areas and red zones representing low demand areas