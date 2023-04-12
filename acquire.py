# acquire.py

# general modules
import numpy as np
import pandas as pd

# local modules
import env  #contains connection variables


def get_connection(db, u=user, h=host, p=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{u}:{p}@{h}/{db}'

def get_zillow_data():
    '''
    This function reads in zillow 2017 data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df. Function relies
    on other functions in the wrangle.py module.
    '''
    if os.path.isfile('zillow_large.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_large.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow_large.csv')
        
    return df

def new_zillow_data():
    '''
    This function queries the zillow database from the CodeUp MySQL server. 
    It retrieves records of Single Family Residential and Inferred Single 
    Family Residential properties that had a transaction in 2017 and imports
    them into a DataFrame. Function relies on other functions in the 
    wrangle.py module.
    
    Arguments: None
    
    Returns: DataFrame of properties queried
    '''
    sql_query = """
                select * from properties_2017 as prop17
	left join airconditioningtype
		using(airconditioningtypeid)
	left join architecturalstyletype
		using(architecturalstyletypeid)
	left join buildingclasstype
		using(buildingclasstypeid)
	left join heatingorsystemtype
		using(heatingorsystemtypeid)
	left join predictions_2016
		using(parcelid)
	left join predictions_2017 as pred17
		using(parcelid)
	left join properties_2016
		using(parcelid)
    left join propertylandusetype as plut
		on prop17.propertylandusetypeid = plut.propertylandusetypeid
	left join storytype as st
		on prop17.storytypeid = st.storytypeid
	left join typeconstructiontype as ct
		on prop17.typeconstructiontypeid = ct.typeconstructiontypeid
	left join unique_properties
		using (parcelid)
	where propertylandusedesc IN (
		'Single Family Residential',
		'Inferred Single Family Residential')
	and 
		pred17.transactiondate between 
			date('2017-01-01') and
			date('2017-12-31')
	;
                 """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df
