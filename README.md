# swappy
Weather Prediction Model Using ML
# Weather Prediction Flask Application

## Overview

This project is a web application built using Flask that predicts the next day's weather based on historical weather data for Kolkata. The predictions are made for three parameters: maximum temperature (`tmax`), minimum temperature (`tmin`), and precipitation (`prcp`). The application utilizes the `meteostat` library to fetch historical weather data and employs Ridge regression models to make predictions.

## Features

- Fetches historical weather data for Kolkata.
- Predicts the next day's maximum temperature, minimum temperature, and precipitation.
- Visualizes the actual weather data for the past 30 days and overlays the predictions.
- Saves the prediction plot as an image and displays it on the web page.

## Requirements

- Python 3.6 or higher
- Flask
- pandas
- meteostat
- scikit-learn
- matplotlib

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/weather-prediction-app.git
    cd weather-prediction-app
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**

    ```sh
    python app.py
    ```

2. **Open your web browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

## File Structure

- `app.py`: The main Flask application script.
- `templates/index.html`: The HTML template for the web page.
- `static/`: Directory for static files, including the plot image.
- `requirements.txt`: List of required Python packages.

## Explanation

### app.py

- **Imports**: Imports necessary libraries including Flask, pandas, datetime, meteostat, scikit-learn, matplotlib, and others.
- **Flask App**: Sets up the Flask application.
- **Route**: Defines the route for the home page (`/`).
- **Weather Data**: Fetches historical weather data for Kolkata using the `meteostat` library.
- **Data Processing**: Processes the weather data, fills missing values, and defines predictors and target columns.
- **Model Training**: Trains Ridge regression models for predicting `tmax`, `tmin`, and `prcp`.
- **Prediction**: Uses the trained models to predict the next day's weather based on the latest available data.
- **Visualization**: Generates a plot showing the actual weather data for the past 30 days and the predictions for the next day.
- **Template Rendering**: Renders the `index.html` template with the prediction results and the plot.

### index.html

- A simple HTML template that displays the weather predictions and the generated plot image.

## Notes

- The application saves the historical weather data to a CSV file with a unique timestamp to avoid overwriting.
- The static directory is created if it does not exist to store the generated plot image.
- The predictions are rounded to two decimal places for display.

