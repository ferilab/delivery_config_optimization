
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def train_delivery_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"\nModel RMSE on test set: {rmse:.2f} minutes")

    # Optionally save model
    joblib.dump(model, package_root + "/models/delivery_time_model.pkl")

    return model
