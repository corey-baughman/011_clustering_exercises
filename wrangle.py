import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from env import host, user, password


# Variables/Iterables
'''
col_list establishes a list of columns with significant outliers as
discovered in univariate analysis. These are primarily right skewed
(very large high-end properties). This model is to predict assessed tax
values of homes, so it seems best to have it perform well on the vast
majority of properties instead of the outliers which, like fine art,
have a much less regular connection to normal market parameters and 
would likely distort the model.
'''
col_list = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount', 'tax_value_2016']

    
# Missing Values:
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


# Finding and Dealing With Outliers

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df

# to view 
# outlier_cols = [col for col in df if col.endswith('_outliers')]
# for col in outlier_cols:
#     print('~~~\n' + col)
#     data = df[col][df[col] > 0]
#     print(data.describe())


def remove_outliers(df, col_list=col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns
    using the tukey method.
    
    Arguments: a DataFrame, col_list=[list of column names or indexes]
                , a k value that equals the number of InterQuartile Ranges
                outside of Q1 and Q3 that will define outliers to be removed.
                col_list defaults to the col_list variable in this module.
                k defaults to the standard 1.5 * IQR for Tukey method.
                
    Returns: a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    df.reset_index(drop=True, inplace=True)
    
    return df

def split_data(df):
    '''
    Take in a DataFrame and return train, validate, and test DataFrames; 
    Return train, validate, test DataFrames. Train gets 56% of observations
    Validate 24% and Test 20%.
    
    Arguments: a DataFrame
    
    Returns: train, validate, test DataFrames
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=9751)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=9751)
    return train, validate, test


# ====================================
# Aggregate Wrangling Functions

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

def clean_zillow_data2017():
    '''
    This function retrieves the zillow data from the CodeUp MySQL database
    and applies cleaning steps to drop observations with null values,
    reset the index after dropping rows, and cast bedroomcnt, yearbuilt, 
    and fips to integers. It adds the features tax_rate and zip_mean_tv. It returns the 
    cleaned dataframe. Function relies on other functions in the wrangle.py module.
    '''
    df = get_zillow_data2017()
    # standardize column names to something more pythonic
    df = df.rename(columns = {'bedroomcnt' : 'bedrooms', 
                              'bathroomcnt' : 'bathrooms', 
                              'calculatedfinishedsquarefeet' :'area', 
                              'taxvaluedollarcnt' : 'tax_value', 
                              'yearbuilt' : 'year_built',  
                              'regionidzip' : 'zip'})
    # dropping all nulls as they are less than 1% of observations
    # and scattered across the features.
    df = df.dropna()
    # may as well reset the index after dropping nulls
    df.reset_index(drop=True, inplace=True)
    # bedrooms, year built, and fips code should be integers
    df.bedrooms = df.bedrooms.astype(int)
    df.year_built = df.year_built.astype(int)
    df.fips = df.fips.astype(int)
    df.zip = df.zip.astype(int)
    df.fips = df.fips.map({6111:'ventura_county', 6059:'orange_county', 6037:'la_county'})
    # add feature 'age'
    df['age'] = 2017 - df.year_built
    # I want to add a feature called tax_rate that I think may be a proxy
    # for location that is more granular than FIPS. (see README.MD for a
    # summary of CA property taxes)
    df['tax_rate_2016'] = round((df.tax_amount / df.tax_value_2016), 5)
    # establish mean tax values for each zip code
    zip_tv = df.tax_value_2016.groupby(df.zip).agg('mean').round().to_dict()
    df['zip_mean_tv_2016'] = df.zip.apply(lambda x: zip_tv[x])
    return df

def wrangle_zillow():
    '''
    This function retrieves the zillow data from the CodeUp MySQL database
    and applies cleaning steps to drop observations with null values,
    resets the index after dropping rows, and cast bedrooms, year_built, 
    and fips to integers. Then it removes outliers using the tukey method
    and finally splits the dataframe into Train, Validate, and Test dataframes. 
    Function relies on other functions in the wrangle.py module.
    
    Arguments: None
    
    Returns: Train, Validate, Test dataframes
    '''
    
    train, validate, test = split_data(remove_outliers(clean_zillow_data2017()))
    
    
    return train, validate, test