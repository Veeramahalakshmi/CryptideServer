from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

# Fetch historical data from CoinGecko
def fetch_historical_data(coin_id, days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    headers = {'accept': 'application/json', 'x-cg-demo-api-key': 'CG-DCauk5sqNi6FCMGCSUqPmmhL'}
    response = requests.get(url, headers=headers)
    data = response.json()
    prices = data['prices']  # List of [timestamp, price]
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['day'] = (df['timestamp'] - df['timestamp'].min()).dt.days  # Days since start
    return df

# Train a simple Linear Regression model and predict future prices
def predict_future_prices(coin_id, days_to_predict=7):
    df = fetch_historical_data(coin_id)
    
    # Prepare data for ML
    X = df[['day']].values
    y = df['price'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future prices
    last_day = df['day'].max()
    future_days = np.array([[last_day + i] for i in range(1, days_to_predict + 1)])
    predictions = model.predict(future_days)
    
    # Generate dates for predictions
    last_date = df['timestamp'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    
    return [{'date': date.strftime('%Y-%m-%d'), 'price': float(price)} for date, price in zip(future_dates, predictions)]

# API endpoint to get predictions
@app.route('/api/predict/<coin_id>', methods=['GET'])
def get_predictions(coin_id):
    try:
        predictions = predict_future_prices(coin_id)
        return jsonify({'coin_id': coin_id, 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)