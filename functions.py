import pandas as pd


def load_data():
    # Load data
    data = pd.read_csv('dataset_mood_smartphone.csv')
    data['time'] = pd.to_datetime(data['time'])
    return data


def remove_incorrect_values(data):
    return data
    # remove rows where appCat.builtin and appCat.entertainment	are negative
    return data[(data['appCat.builtin'] >= 0) & (data['appCat.entertainment'] >= 0)]
    # DOES NOT WORK YET FOR SOME REASON!!!!!!!!!


def replace_missing_long(data, id_only=False):
    if id_only:
        # replace missing values with the mean of the variable for that id
        data['value'] = data.groupby(['id', 'variable'])[
            'value'].transform(lambda x: x.fillna(x.mean()))
    else:
        # replace missing values with the mean of the variable for that id and day
        data['value'] = data.groupby(['id', 'variable', data['time'].dt.date])[
            'value'].transform(lambda x: x.fillna(x.mean()))
    return data


def widen_data(data):
    # Widen data
    data = data.pivot_table(
        index=['id', 'time'], columns='variable', values='value').reset_index()
    return data


def group_data(data_wide):
    # sum variables
    sum_vars = ['appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other',
                'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities',
                'appCat.weather', 'screen']
    # mean variables
    mean_vars = ['circumplex.arousal',
                 'circumplex.valence', 'activity', 'mood']

    # group the wide data by day and id and aggregate the sum and mean of the variables
    return data_wide.groupby(pd.Grouper(key='time', freq='D')).agg({**{var: 'sum' for var in sum_vars}, **{var: 'mean' for var in mean_vars}})


def replace_missing_wide(data):
    # TODO
    return data


def clean_data(data=load_data()):
    data = remove_incorrect_values(data)
    data = replace_missing_long(data)
    data = widen_data(data)
    data = group_data(data)
    data = replace_missing_wide(data)
    return data


def iqr(data):
    if pd.api.types.is_numeric_dtype(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data > lower_bound) & (data < upper_bound)]
    else:
        return data


def remove_outliers(data_wide):
    return data_wide.apply(iqr)


def normalize_data(data):
    to_normalize = ['circumplex.arousal',
                    'circumplex.valence', 'mood', 'activity', 'screen']
    # Normalize data
    # TODO


# 1C FEATURE ENGINEERING
def feature_engineering(data_wide):
    # Extract hour and day information from the 'time' column
    data_wide['hour'] = data_wide['time'].dt.hour
    data_wide['day'] = data_wide['time'].dt.dayofweek
    return data_wide
