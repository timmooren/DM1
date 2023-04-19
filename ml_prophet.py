import pandas as pd
from prophet import Prophet
from functions import clean_data


def main(data):
    # Prepare the data
    df_prophet = data.rename(columns={'time': 'ds', 'mood': 'y'})

    # Initialize the model
    model = Prophet()

    # Fit the model on the dataset
    model.fit(df_prophet)

    # Make future predictions
    future = model.make_future_dataframe(
        periods=1)  # predict the next 'periods' days
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    # save the plot
    fig.savefig('forecast.png')

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


if __name__ == '__main__':
    data = clean_data()
    main(data)
