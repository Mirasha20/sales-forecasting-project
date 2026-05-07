from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

# Load model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "Sales Forecast API Running"}

@app.get("/predict")
def predict():

    try:
        # Read Excel file
        df = pd.read_csv("forecast_input.csv.xls")

        # Show available columns
        print("Columns:", df.columns.tolist())

        # Required columns
        features = [
            'lag_1',
            'lag_7',
            'lag_30',
            'rolling_mean',
            'rolling_std',
            'day_of_week',
            'month'
        ]

        # Keep only required columns
        df = df[features]

        predictions = model.predict(df)

        return {
            "forecast": predictions.tolist()
        }

    except Exception as e:
        return {
            "error": str(e)
        }