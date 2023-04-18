import pandas as pd
from scipy.stats import boxcox

def load_data():
    # Load data
    data = pd.read_csv('dataset_mood_smartphone.csv')
    data['time'] = pd.to_datetime(data['time'])
    return data


def remove_incorrect_values(data):
    # makes copy of df to not modify argument 
    data_copy = data.copy()
    # remove rows where appCat.builtin and appCat.entertainment	are negative
    return data_copy[~(((data_copy['variable'] == 'appCat.builtin') | (data_copy['variable'] == 'appCat.entertainment')) & (data_copy['value'] < 0))]


def replace_missing_long(data, id_only=True):
    # makes copy of df to not modify argument 
    data_copy = data.copy()
    if id_only:
        # replace missing values with the mean of the variable for that id
        data_copy['value'] = data_copy.groupby(['id', 'variable'])[
            'value'].transform(lambda x: x.fillna(x.mean()))
    else:
        # replace missing values with the mean of the variable for that id and day
        # NOTE there are days with NaN values only; mean cannot be calculated for that specific day
        data_copy['value'] = data_copy.groupby(['id', 'variable', data_copy['time'].dt.date])[
            'value'].transform(lambda x: x.fillna(x.mean()))
    return data_copy


def widen_data(data):
    # makes copy of df to not modify argument 
    data_copy = data.copy()
    # Widen data
    data_copy = data_copy.pivot_table(
        index=['id', 'time'], columns='variable', values='value').reset_index()
    return data_copy


def group_data(data_wide):
    # makes copy of df to not modify argument 
    data_wide_copy = data_wide.copy()
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
    data_wide_copy[sum_vars] = data_wide_copy[sum_vars].fillna(0)

    # old aggregation method 
    # data_wide_copy = data_wide_copy.groupby(
    #     [pd.Grouper(key='time', freq='D'), 'id']).mean().reset_index()

    # 1st aggregation - group by day while maintaining individuals (sum & mean)
    data_wide_copy = data_wide_copy.groupby([pd.Grouper(key='time', freq='D'), 'id']).agg({**{var: 'sum' for var in sum_vars}, **{var: 'mean' for var in mean_vars}}).reset_index()

    # 2nd aggregation - group individuals together (only mean)
    data_wide_copy = data_wide_copy.groupby(
        [pd.Grouper(key='time', freq='D')]).mean().reset_index()
    
    count_df = data_wide.groupby([pd.Grouper(key='time', freq='D'), 'id']).count().reset_index() # {**{var: 'sum' for var in sum_vars}, **{var: 'mean' for var in mean_vars}}
    count_df = count_df.groupby(pd.Grouper(key='time', freq='D')).mean()

    result = pd.merge(data_wide_copy, count_df, on='time', suffixes=[None,'_count'])
    return result


def impute_missing_wide(data):
    # makes copy of df to not modify argument 
    data_copy = data.copy()
    # removes first 14 days and last day (too much missing data)
    data_copy = data_copy[15:-1]
    # imputation methods 
    data_copy['activity'] = data_copy['activity'].bfill()
    data_copy[['circumplex.arousal','circumplex.valence', 'mood']] = data_copy[['circumplex.arousal','circumplex.valence', 'mood']].interpolate(method='linear')

    return data_copy


def clean_data(data=load_data()):
    data = remove_incorrect_values(data)
    # HERE remove outliers
    data = replace_missing_long(data)
    data = widen_data(data)
    data = group_data(data)
    data = impute_missing_wide(data)
    
    return data


def iqr(data):
    outliers = []
    for var in data['variable'].unique():
        
        partial = data.loc[data['variable'] == var]
        values = data.loc[data['variable'] == var]['value']
        min = values.min()
        if min < 0:
            add = -min + 0.001
            values += add
        if min == 0:
            add = 0.001
            values += add
        
        if var != 'sms' and var != 'call':
            partial['transformed'] = boxcox(values)[0]
            Q1 = partial['transformed'].quantile(0.25)
            Q3 = partial['transformed'].quantile(0.75)
            IQR = Q3 - Q1
            # use 3 for extreme outliers
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            upper = partial[(partial['transformed'] > upper_bound)]
            lower = partial[(partial['transformed'] < lower_bound)] # 
            
            if len(upper) != 0:
                outliers.append(upper)
            if len(lower != 0):
                outliers.append(lower)
            
    return outliers


def remove_outliers(data_wide):
    #return data_wide.apply(iqr)
    pass


def normalize_data(data):
    # Normalize data
    data_scaled = data.copy()
  
    # apply normalization techniques
    for column in [col for col in data.columns if col != 'time']:
        data_scaled[column] = (data_scaled[column] - data_scaled[column].min()) / (data_scaled[column].max() - data_scaled[column].min())    
    return data_scaled


# 1C FEATURE ENGINEERING
def feature_engineering(data_wide):
    # Extract hour and day information from the 'time' column
    # data_wide['hour'] = data_wide['time'].dt.hour
    data_wide['day'] = data_wide['time'].dt.dayofweek
    return data_wide
