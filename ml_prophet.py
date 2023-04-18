import pandas as pd
from prophet import Prophet



def temp(data):
    # Prepare the data
    data['time'] = pd.to_datetime(data['time'])
    df_prophet = df_prophet.rename(columns={'time': 'ds', 'mood': 'y'})

    # Initialize the model
    model = Prophet()

    # Fit the model on the dataset
    model.fit(df_prophet)

    # Make future predictions
    future = model.make_future_dataframe(periods=1)  # predict the next 'periods' days
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
