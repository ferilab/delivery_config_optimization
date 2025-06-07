
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings("ignore")

def recommend_optimal_config(context_dict, model, encoder):
    # Controllable variables to simulate
    vehicles = ['Bike', 'Car', 'Scooter']  # Adapt based on actual values in dataset
    agent_ages = list(range(20, 60, 5))    # Simulate reasonable agent ages
    agent_ratings = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Simulate rating levels

    # Now, we'll make all possible combination of the controlable variables.
    configs = list(product(vehicles, agent_ages, agent_ratings))
    simulated_df = pd.DataFrame(configs, columns=['Vehicle', 'Agent_Age', 'Agent_Rating'])

    # Then, we'll add fixed context (here, hypothetic) to all rows (each row is a combination of controlable variables)
    for col, val in context_dict.items():
        simulated_df[col] = val

    # Encode categorical variables using the encoder (that already fitted to the train data)
    cat_cols = ['Vehicle', 'Weather', 'Traffic', 'Area', 'Category', 'DayOfWeek']
    encoded = encoder.transform(simulated_df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=simulated_df.index)

    model_input = pd.concat([simulated_df.drop(columns=cat_cols), encoded_df], axis=1)

    # Predict delivery time
    simulated_df['Predicted_Delivery_Time'] = model.predict(model_input)

    # Return config with lowest predicted time
    best_row = simulated_df.sort_values('Predicted_Delivery_Time').iloc[0]
    return best_row[['Vehicle', 'Agent_Age', 'Agent_Rating', 'Predicted_Delivery_Time']]
