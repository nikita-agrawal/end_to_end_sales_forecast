#BATCH INFERENCE SCRIPT FOR FORECASTING SALES DEMAND

import mlflow
import mlflow.xgboost
import datetime 
import pandas as pd 
import xgboost as xgb

def load_latest_model(model_name, alias):
    # Load the latest version of the model  
    uri = f"models:/{model_name}@{alias}"
    latest_model = mlflow.xgboost.load_model(uri)
    return latest_model

def generate_test_data(today, forecast_window = 30):
    forecast_dates = [today + datetime.timedelta(days=i) for i in range(forecast_window)]
    test_data = pd.DataFrame(forecast_dates, columns=['forecast_date'])
    
    # Convert forecast_date to datetime format
    test_data['forecast_date'] = pd.to_datetime(test_data['forecast_date'])
    test_data['dayofweek'] = test_data['forecast_date'].dt.dayofweek
    test_data['quarter'] = test_data['forecast_date'].dt.quarter
    test_data['month'] = test_data['forecast_date'].dt.month
    test_data['year'] = test_data['forecast_date'].dt.year
    test_data['dayofyear'] = test_data['forecast_date'].dt.dayofyear
    test_data['dayofmonth'] = test_data['forecast_date'].dt.day
    test_data['weekofyear'] = test_data['forecast_date'].dt.isocalendar().week
    return test_data

def run_batch_inference():
    # Set starting date for which to forecast sales demand
    today = datetime.datetime.today().date()
    # Load the model
    model_name = 'xgboost-model'
    alias = 'latest_model'
    model = load_latest_model(model_name, alias)

    #Print info about model used for inference
    client = mlflow.tracking.MlflowClient()
    model_version_info = client.get_registered_model(model_name).latest_versions[0]
    print(f"------Step 1------: Loaded latest trained model: '{model_name}', run_id: '{model_version_info.run_id}, version: '{model_version_info.version}, alias: 'latest_model'")
    
    # Load the data
    #today = datetime.datetime.today().date() # if live data  
    test_data = generate_test_data(today, 30)
    print(f"------Step 2------: Generated test data for inference: start date: '{today}', forecast window: '30 days'")


    # Perform batch inference
    # Convert the Pandas DataFrame to XGBoost DMatrix
    dmatrix_input = xgb.DMatrix(test_data.drop(['forecast_date'], axis=1))
    # Make predictions using the model
    predictions = model.predict(dmatrix_input)
    print(f"------Step 3------: Completed batch inference for '{len(test_data)}' records")

    # Format predictions
    save_data = pd.DataFrame({
    'run_date': today,
    'forecast_date': test_data['forecast_date'],
    'forecasted_sales_demand': predictions })

    # Save the predictions to a CSV file or database 
    csv_path = f"/Users/nikiagrawal/Desktop/Python_Dev/output_database/sales_forecast_{today.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    save_data.to_csv(csv_path, index=False)
    print(f"------Step 4------: Saved predictions to CSV.file:{csv_path}")

# Main function to run the forecast
if __name__ == "__main__":
    print(f"Starting batch inference at {datetime.datetime.now()}...")
    run_batch_inference()

    