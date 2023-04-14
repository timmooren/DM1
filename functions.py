import pandas as pd


def load_data():
    # Load data
    data = pd.read_csv('dataset_mood_smartphone.csv')
    data['time'] = pd.to_datetime(data['time'])
    return data


def remove_incorrect_values(data):
    # remove rows where appCat.builtin and appCat.entertainment	are negative
    return data[~(((data['variable'] == 'appCat.builtin') | (data['variable'] == 'appCat.entertainment')) & (data['value'] < 0))]


def replace_missing_long(data, id_only=True):
    if id_only:
        # replace missing values with the mean of the variable for that id
        data['value'] = data.groupby(['id', 'variable'])[
            'value'].transform(lambda x: x.fillna(x.mean()))
    else:
        # replace missing values with the mean of the variable for that id and day
        # NOTE there are days with NaN values only; mean cannot be calculated for that specific day
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
                'appCat.weather', 'screen', 'call', 'sms']
    # mean variables
    mean_vars = ['circumplex.arousal',
                 'circumplex.valence', 'activity', 'mood']
    mean_vars = sum_vars + mean_vars

    # for the sum variables, replace nan values with 0
    data_wide[sum_vars] = data_wide[sum_vars].fillna(0)

    # group the wide data by day and id and aggregate the mean of the variables
    data_wide = data_wide.groupby(
        [pd.Grouper(key='time', freq='D'), 'id']).mean().reset_index()

    # group the wide data by day and mean
    data_wide = data_wide.groupby(
        [pd.Grouper(key='time', freq='D')]).mean().reset_index()
    return data_wide


def replace_missing_wide(data):
    # TODO
    return data


def clean_data(data=load_data()):
    data = remove_incorrect_values(data)
    # HERE remove outliers
    data = replace_missing_long(data)
    data = widen_data(data)
    data = group_data(data)
    data = replace_missing_wide(data)
    return data


def iqr(data):
    for var in data['variable'].unique():
        partial = data.loc[data['variable'] == var]['value']
        if pd.api.types.is_numeric_dtype(partial):
            Q1 = partial.quantile(0.25)
            Q3 = partial.quantile(0.75)
            IQR = Q3 - Q1
            # use 3 for extreme outliers
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            data = data[~((data['variable'] == var) & (
                (data['value'] < lower_bound) | (data['value'] > upper_bound)))]
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
