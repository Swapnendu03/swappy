from flask import Flask, render_template
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def index():
    # Define the location and time period
    location = Point(55.7558, 37.6173)  # Coordinates for Kolkata
    start = datetime(2021, 5, 27)
    end = datetime(2024, 5, 29)

    # Fetch the historical weather data
    data = Daily(location, start, end)
    data = data.fetch()

    # Create a timestamp for the file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Define the file name with the timestamp
    file_name = f'historical_weather_data_{timestamp}.csv'

    # Save the data to a CSV file with the unique file name
    data.to_csv(file_name)

    # Load the weather data from the CSV file
    weather = pd.read_csv(file_name, index_col="time")
    weather.index = pd.to_datetime(weather.index)  # Ensure the index is datetime
    weather = weather.fillna(0)  # Ensure no NaN values

    # Define the target columns
    target_columns = ['tmax', 'tmin', 'prcp']

    # Define the predictors (features), excluding the target columns
    predictors = weather.columns.difference(target_columns).tolist()

    # Train separate models for tmax, tmin, and prcp
    model_tmax = Ridge(alpha=0.1)
    model_tmin = Ridge(alpha=0.1)
    model_prcp = Ridge(alpha=0.1)

    # Train the models on the entire dataset
    model_tmax.fit(weather[predictors], weather["tmax"])
    model_tmin.fit(weather[predictors], weather["tmin"])
    model_prcp.fit(weather[predictors], weather["prcp"])

    # Get today's weather data
    today_data = weather.iloc[-1][predictors].values.reshape(1, -1)

    # Predict tomorrow's tmax
    tomorrow_tmax = model_tmax.predict(today_data)[0]

    # Prepare data for predicting tmin
    today_data_tmin = weather.iloc[-1][predictors + ['tmax']].copy()
    today_data_tmin['tmax'] = tomorrow_tmax
    today_data_tmin = today_data_tmin[predictors].values.reshape(1, -1)
    tomorrow_tmin = model_tmin.predict(today_data_tmin)[0]

    # Prepare data for predicting prcp
    today_data_prcp = weather.iloc[-1][predictors + ['tmax', 'tmin']].copy()
    today_data_prcp['tmax'] = tomorrow_tmax
    today_data_prcp['tmin'] = tomorrow_tmin
    today_data_prcp = today_data_prcp[predictors].values.reshape(1, -1)
    tomorrow_prcp = model_prcp.predict(today_data_prcp)[0]

    # Round off the predictions
    tomorrow_tmax = round(tomorrow_tmax, 2)
    tomorrow_tmin = round(tomorrow_tmin, 2)
    tomorrow_prcp = round(tomorrow_prcp, 2)

    # Visualize today's actual weather and tomorrow's predicted weather
    plt.figure(figsize=(10, 6))
    plt.plot(weather.index[-30:], weather['tmax'][-30:], label='Actual tmax (last 30 days)', marker='o')
    plt.axvline(weather.index[-1], color='gray', linestyle='--')
    plt.plot([weather.index[-1], weather.index[-1] + pd.Timedelta(days=1)], [weather['tmax'].iloc[-1], tomorrow_tmax],
             'ro-', label="Predicted tmax")
    plt.text(weather.index[-1] + pd.Timedelta(days=1), tomorrow_tmax, f'{tomorrow_tmax:.2f}',
             color='red', ha='left')

    plt.plot(weather.index[-30:], weather['tmin'][-30:], label='Actual tmin (last 30 days)', marker='o')
    plt.plot([weather.index[-1], weather.index[-1] + pd.Timedelta(days=1)], [weather['tmin'].iloc[-1], tomorrow_tmin],
             'bo-', label="Predicted tmin")
    plt.text(weather.index[-1] + pd.Timedelta(days=1), tomorrow_tmin, f'{tomorrow_tmin:.2f}',
             color='blue', ha='left')

    plt.plot(weather.index[-30:], weather['prcp'][-30:], label='Actual prcp (last 30 days)', marker='o')
    plt.plot([weather.index[-1], weather.index[-1] + pd.Timedelta(days=1)], [weather['prcp'].iloc[-1], tomorrow_prcp],
             'go-', label="Predicted prcp")
    plt.text(weather.index[-1] + pd.Timedelta(days=1), tomorrow_prcp, f'{tomorrow_prcp:.2f}',
             color='green', ha='left')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title("Today's Actual Weather and Tomorrow's Predicted Weather")
    plt.legend()
    plt.grid(True)

    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Save the plot to a file
    plot_file = os.path.join('static', 'plot.png')
    plt.savefig(plot_file)
    plt.close()

    return render_template('index.html', plot_url=plot_file, tmax=tomorrow_tmax, tmin=tomorrow_tmin, prcp=tomorrow_prcp)

if __name__ == '__main__':
    app.run(debug=True)
