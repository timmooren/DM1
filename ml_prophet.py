import pandas as pd
from prophet import Prophet
from functions import clean_data
from sklearn.model_selection import ParameterGrid
from prophet.diagnostics import cross_validation, performance_metrics


def main(data):
    # Prepare the data
    df_prophet = data.rename(columns={'time': 'ds', 'screen': 'y'})

    # Define the hyperparameters to tune
    param_grid = {
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.9, 1.0],
        'seasonality_prior_scale': [0.01, 0.1, 1.0]
    }

    # Initialize the best model and its performance
    best_model = None
    best_performance = float('inf')
    metric = 'mae'

    # Iterate over all hyperparameter combinations
    for params in ParameterGrid(param_grid):
        # Initialize the model with the current hyperparameters
        model = Prophet(
            seasonality_mode=params['seasonality_mode'],
            changepoint_range=params['changepoint_range']
        )

        # Add regularization parameters separately
        model.add_seasonality(
            name='daily',
            period=1,
            fourier_order=10,
            prior_scale=params['seasonality_prior_scale']
        )

        # Fit the model on the dataset
        model.fit(df_prophet)

        # Cross-validate the model
        cv_results = cross_validation(model, horizon='1 day')

        # Compute the mean squared error (MSE) performance metric
        performance_metric = performance_metrics(cv_results)['mae'].mean()
        performance_metric2 = performance_metrics(cv_results)['mse'].mean()

        # Check if this model is better than the previous best model
        if performance_metric < best_performance:
            best_model = model
            best_performance = performance_metric


    # Print the best hyperparameters and performance
    print("Best hyperparameters:", best_model.params)
    print("Best performance mae:", best_performance)

    # Make future predictions
    future = best_model.make_future_dataframe(
        periods=1)  # predict the next 'periods' days
    forecast = best_model.predict(future)

    # Plot the forecast
    fig = best_model.plot(forecast)
    # save the plot
    fig.savefig(f'plots/forecast_{metric}.png')

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


if __name__ == '__main__':
    data = clean_data()
    main(data)
