# Import necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# # Register matplotlib converters to be able to plot timestamps
pd.plotting.register_matplotlib_converters()

# Read the dataset
file_path = './dataset/petrol_price.csv'  # Updated to the correct file path
petrol_price_data = pd.read_csv(file_path)

# Convert 'ds' to datetime format mixed
petrol_price_data['ds'] = pd.to_datetime(petrol_price_data['ds'], format='mixed')

# Format the dates in 'YYYY-MM-DD' format
petrol_price_data['ds'] = petrol_price_data['ds'].dt.strftime('%Y-%m-%d')

# print the first 30 rows of the petrol_price_data
# print(petrol_price_data.head(30))

# Define the model fitting and prediction function
def fit_predict_model(dataframe, interval_width=0.99, changepoint_range=0.8):
    # Initialize the Prophet model with specified parameters
    model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
                seasonality_mode='multiplicative', 
                interval_width=interval_width,
                changepoint_range=changepoint_range)
    
    # Fit the model to the data
    model = model.fit(dataframe)
    
    # Create a DataFrame for future predictions
    future = model.make_future_dataframe(periods=365)
    
    # Predict future values
    forecast = model.predict(future)
    
    # Plot the forecast
    model.plot(forecast)
    plt.show()  # Display the plot
    
    return forecast

# Fit the model and make a forecast
petrol_price_data.rename(columns={'ds': 'ds', 'y': 'y'}, inplace=True)  # Ensure the column names match expected by Prophet
forecast = fit_predict_model(petrol_price_data)

# Define the anomaly detection function
def detect_anomalies(forecast):
    forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    
    # Initialize 'anomaly' column as integer and 'importance' column as float
    forecasted['anomaly'] = 0
    forecasted['importance'] = 0.0  # Explicitly set as float

    # Find positive anomalies (actual values above the upper confidence interval)
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    
    # Find negative anomalies (actual values below the lower confidence interval)
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    # Calculate the importance of positive anomalies
    forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper']) / forecasted['fact']
    
    # Calculate the importance of negative anomalies
    forecasted.loc[forecasted['anomaly'] == -1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact']) / forecasted['fact']
    
    return forecasted

# Detect anomalies in the forecast
forecast['fact'] = petrol_price_data['y'].reset_index(drop=True)  # This line is added to ensure we have actual values to compare with
prediction = detect_anomalies(forecast)

# Plot the results with actual values, predicted values, and confidence intervals
prediction.plot(x='ds', y=['fact', 'yhat', 'yhat_upper', 'yhat_lower'], figsize=(14, 7))
plt.title('Petrol Price Over Time')
plt.xlabel('ds')
plt.ylabel('Price')
plt.show()

# Filter and display data where an anomaly was detected
anomaly_data = prediction[prediction['anomaly'] != 0]  # This should be != 0 to include both 1 and -1 anomalies
print(anomaly_data.head())
