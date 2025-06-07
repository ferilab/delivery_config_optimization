# delivery_config_optimization
Using an Amazon delivery dataset, a delivery configuration recommender is developed to minimize delivery time (target) by choosing best driver configurations. It considers contextual factors (like weather, traffic, and area) to offer optimum combination of controllable variables (like vehicle and agent's rating).

# Dataset
https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset

This Amazon Delivery Dataset provides a comprehensive view of the company's last-mile logistics operations. It includes data on over 43,632 deliveries across multiple cities, with detailed information on order details, delivery agents, weather and traffic conditions, and delivery performance metrics. The dataset enables researchers and analysts to uncover insights into factors influencing delivery efficiency, identify areas for optimization, and explore the impact of various variables on the overall customer experience.

The Amazon Delivery Dataset is provided as a single CSV file named "amazon_delivery.csv". The file contains 43,632 rows of data, representing individual delivery records, and 16 columns as described above.
The file size is approximately 6 MB that is organized in a tabular format, with each row representing a delivery and each column representing a specific attribute or feature. The CSV file uses commas as the delimiter and includes a header row that lists the column names. The file is encoded in UTF-8 format, ensuring compatibility with most text editors and programming environments.

The columns are:
Order_ID
Agent_Age
Agent_Rating
Store_Latitude
Store_Longitude
Drop_Latitude
Drop_Longitude
Order_Date
Order_Time
Pickup_Time
Weather
Traffic
Vehicle
Area
Delivery_Time
Category


The optimization strategy:

1. Input variables (conrollable):
   
While Agent_Age may indirectly reflect experience, Agent_Rating is a more direct performance signal. Therefore, including both gives the model a better sense of agent reliability and service quality.
Also Vehicle is an important independent variable here.

2. Contextual (out of control):

Geo coordinates (Store_Latitude, etc.)
Area
Temporal features (Order_date, Order_Time, Pichup_Time). We expect the day of week, earlier or later time of the day, and even season affects the delivery time.
Weather
Traffic
Category

3. Target:
   
Delivery_Time



The package structure:

delivery_config_optimization/
│
├── data/
│   └── amazon_delivery.csv       # Raw dataset from Kaggle
│
├── dev_run/
│   └── delivery_config_opt.ipynb    # EDA + model training notebook
│
├── src/
│   ├── prepare_data.py           # Clean and encode data
│   ├── train_model.py            # Train regression model
│   └── optimize_conditions.py    # Recommend best delivery config
|   └── __init__.py    
│
├── models/
│   └── delivery_time_model.pkl   # This will be created and save by the code (optional)
│
├── main.py                       # Script (can be CLI too) to run full pipeline
├── requirements.txt              # Dependencies
└── README.md                     # Project overview


prepare_data.py
1. Cleans and filters data
2. Extracts:
    Temporal features (weekday, hour, pickup delay)
    Distance using the Haversine formula
3. One-hot encodes all categorical variables
4. Returns clean feature matrix, target column, and encoder

train_model.py
1. Trains a GradientBoostingRegressor on delivery data
2. Evaluates with RMSE
3. Saves the model to models/delivery_time_model.pkl (optional)
4. optimize_conditions.py — searches for the best combo of Vehicle + Agent_Age + Agent_rate under fixed contextual inputs.

optimize_conditions.py
1. Simulates all combinations of Vehicle, Agent_Age, and Agent_Rating
2. Merges them with a given context (e.g., Weather, Traffic, etc.)
3. Uses the trained model to predict delivery times
4. Returns the configuration that minimizes delivery time

main.py — entry point for running everything.
