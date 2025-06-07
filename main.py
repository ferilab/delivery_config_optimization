
from halo import Halo
import warnings
warnings.filterwarnings("ignore")

from src.prepare_data import load_and_prepare_data
from src.train_model import train_delivery_model
from src.optimize_conditions import recommend_optimal_config

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    spinner = Halo(text='Loading data', spinner='dots') # You can choose various spinner styles
    spinner.start()
    df, feature_cols, target_col, encoder = load_and_prepare_data("data/amazon_delivery.csv")
    spinner.succeed('Data loading done!')

    # Step 2: Train model
    spinner = Halo(text='Training model', spinner='dots') # You can choose various spinner styles
    spinner.start()
    model = train_delivery_model(df[feature_cols], df[target_col])
    spinner.succeed('Model training done!')

    # Step 3: Simulate optimization under given context
    sample_context = {
        "Weather": "Sunny",
        "Traffic": "Medium",
        "Area": "Urban",
        "Category": "Electronics",
        "DayOfWeek": "Tuesday",
        "Order_Hour": 10,
        "Pickup_Delay_Minutes": 20,
        "Distance_km": 5.0
    }

    best_config = recommend_optimal_config(sample_context, model, encoder)
    print("\nRecommended Delivery Configuration:")
    print("-" * 50)
    print(best_config)
