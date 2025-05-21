
import os
import pandas as pd
import numpy as np
from functools import lru_cache
import mmap
import os




def list_files():
    directory = r'/home/workspace/files/'
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('num.csv')]
    
    # Sort files by size in descending order
    csv_files.sort(key=lambda f: os.path.getsize(os.path.join(directory, f)), reverse=True)
    
    
    return csv_files


def filter_and_clean_columns(df):
        # Step 2: Drop unnecessary columns 
    df = df.drop_duplicates()
    df = df.drop(columns=['monitor_id', 'monitor_name', 'sesId', 'key', 'SERVER_TIME','nnPoxy','ncvpMean', 'nnResp', 'nnPulse','ntempNa','ntempRe'])#drop these physiological and temp columns ans less then 0.5% patients have even 1 reading for these parameters
   
    # List of columns you want to force to 'float64'
    float_columns = ['nPleth', 'necgRate','nacooMin', 'nCoo', 
                     'ntempEs', 'ntempSk', 'ntempCo', 'nTemp', 
                     'nawRr', 'nrespRate', 'nartMean', 'nartDia', 
                     'nartSys', 'nabpMean', 'nabpDia', 'nabpSys',
                     'nnbpMean', 'nnbpDia', 'nnbpSys']


    # Step 3: Clean the problematic columns by vectorized conversion to numeric
    df[float_columns] = df[float_columns].apply(pd.to_numeric, errors='coerce')
    
    #Step 4: Drop rows with all NaN values in float columns to to remove dead 
    # space at beginning and end, this will also remove rows later on with no values but will 
    # be imputed back on time resabpling to 1 second for time series purposes. this basically
    # removes dead space at beginning where machine turned on but before connected to patient
    df = df.dropna(subset=float_columns, how='all')
    return df


def process_time_intervals(df):
    
    #Step 1: convert time to seconds and create emty rows for missing seconds   
    # Convert 'devTime' to datetime
    df['devTime'] = pd.to_datetime(df['devTime'], unit='s')
    # Set minimum devTime to zero and adjust all times accordingly
    min_time = df['devTime'].min()
    df['devTime'] = df['devTime'] - min_time  # Adjust all times to be relative to min_time
    # Return the processed DataFrame
    
    df.set_index('devTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]#handles exceedingly rare cases where two entries at same second
    df = df.resample('1S').asfreq()
    df.reset_index(inplace=True)
    #df['patient_id'] = df['patient_id'][0]

    return df
    

def remove_physiologically_implausible_values(df, min_hr = 30, max_hr = 300,max_etco2 = 15):

    # Step 1: filter physiologically implausible values
    df['necgRate'] = np.where((df['necgRate'] > max_hr) | (df['necgRate'] < min_hr), np.nan, df['necgRate'])
    # Assuming your DataFrame is called df
    df.loc[df['nacooMin'] > max_etco2, ['nacooMin', 'nawRr']] = np.nan
    df.loc[df['nCoo'] > max_etco2, ['nCoo', 'nawRr']] = np.nan
    
    
    # For the specified temperature columns, set values to np.nan if they are <25.0 or >45.0
    temp_columns = ['ntempEs', 'ntempSk', 'ntempCo', 'nTemp']
    for col in temp_columns:
        df.loc[(df[col] < 25.0) | (df[col] > 45.0), col] = np.nan
        
   
    # Set values less than 5 in those Blood pressure columns to NaN
    columns_to_update = df.filter(regex='Mean|Sys|Dia').columns   
    df.loc[:, columns_to_update] = df.loc[:, columns_to_update].where(df.loc[:, columns_to_update] >= 5, np.nan)

    return df


def process_bp_columns(df):
    # Define the pairs of columns to merge
    columns_to_merge = [
        ('nartMean', 'nabpMean'),
        ('nartDia', 'nabpDia'),
        ('nartSys', 'nabpSys')
    ]

    for col_primary, col_secondary in columns_to_merge:
        # Merge the columns by taking non-NaN values from the primary,
        # and filling NaNs with values from the secondary column
        df[col_primary] = df[col_primary].combine_first(df[col_secondary])

        # Drop the secondary column as it's no longer needed
        df.drop(columns=[col_secondary], inplace=True)

    return df



def add_shock_index_column(dataframe):
    # Create a copy of the DataFrame to avoid modifying the original
    df_with_shock_index = dataframe.copy()

    # Calculate shock index using vectorized operations
    df_with_shock_index['nshock_index'] = np.where(
        pd.notna(df_with_shock_index['necgRate']) & pd.notna(df_with_shock_index['nartSys']) & (df_with_shock_index['nartSys'] != 0),
        df_with_shock_index['necgRate'] / df_with_shock_index['nartSys'],
        np.where(
            pd.notna(df_with_shock_index['necgRate']) & pd.notna(df_with_shock_index['nnbpSys']) & (df_with_shock_index['nnbpSys'] != 0),
            df_with_shock_index['necgRate'] / df_with_shock_index['nnbpSys'],
            np.nan
        )
    )

    return df_with_shock_index





def calculate_z_score(value, median, std_dev):
    return (value - median) / std_dev


def scale_values(df):
    ##################################################################################################
    #load demographics and extract relevant row
    demo = pd.read_csv('/home/workspace/files/cats_data_extract_20210920_master_deid_20211019.csv')    
    cats_id = df['patient_id'][0]
    patient_row = demo[demo['cats_id'] == cats_id].copy()

    
    #########################################################################################################
    #extract_patient age
    patient_row['age_at_admission'] = patient_row['age_at_admission'] * 12  # Convert years to months
    patient_row['age_months'] = patient_row['age_at_admission'].combine_first(patient_row['age_mon'])
    patient_age = patient_row['age_months']/12#to get age in years
        # Extract scalar value for patient_age in years
    patient_age = patient_age.iloc[0]  # Extract scalar using .iloc[0]
   
   ###################################################################################### 
    #extract_patient gender/sex
    gender_to_sex_mapping = {'Male': 'M', 'Female': 'F'}
    # Step 1: Map 'gender' to 'sex' format and fill NaNs
    mapped_gender = patient_row['gender'].map(gender_to_sex_mapping)
    patient_row['sex'] = patient_row['sex'].combine_first(mapped_gender)
    patient_row['gender'] = patient_row['gender'].combine_first(patient_row['sex'].map({v: k for k, v in gender_to_sex_mapping.items()}))
    # Step 2: Create unified column based on matching conditions
    patient_row['unified'] = np.where(
        (patient_row['gender'].map(gender_to_sex_mapping) == patient_row['sex']),  # Check if gender and sex match
        patient_row['gender'].map({'Male': 'M', 'Female': 'F'}),              # Encode Male as 0, Female as 1
        np.random.choice(['M', 'F'])                                           # in rare case where there is a mismatch between gender and sex columnns, randomly pick M or F. 
                                                                                #this is not super significant as ranges accross M vs F vary only slightly
    )
    patient_sex = patient_row['unified']
    patient_sex = patient_sex.iloc[0]  # Extract scalar using .iloc[0]
   

  ####################################################################################################################### 
    ##z-normalise HR
    age_ranges = {
        "0-0.25": {
            "median": 143,
            "percentile_90": 164,
            "std_dev": 12.76595744680851
        },
        "0.25-0.5": {
            "median": 140,
            "percentile_90": 159,
            "std_dev": 11.55015197568389
        },
        "0.5-0.75": {
            "median": 134,
            "percentile_90": 152,
            "std_dev": 10.94224924012158
        },
        "0.75-1": {
            "median": 128,
            "percentile_90": 145,
            "std_dev": 10.33434650455927
        },
        "1-1.5": {
            "median": 123,
            "percentile_90": 140,
            "std_dev": 10.33434650455927
        },
        "1.5-2": {
            "median": 116,
            "percentile_90": 135,
            "std_dev": 11.55015197568389
        },
        "2-3": {
            "median": 110,
            "percentile_90": 128,
            "std_dev": 10.94224924012158
        },
        "3-4": {
            "median": 104,
            "percentile_90": 123,
            "std_dev": 11.55015197568389
        },
        "4-6": {
            "median": 98,
            "percentile_90": 117,
            "std_dev": 11.55015197568389
        },
        "6-8": {
            "median": 91,
            "percentile_90": 111,
            "std_dev": 12.1580547112462
        },
        "8-12": {
            "median": 84,
            "percentile_90": 103,
            "std_dev": 11.55015197568389
        },
        "12-15": {
            "median": 78,
            "percentile_90": 96,
            "std_dev": 10.94224924012158
        },
        "15-18": {
            "median": 73,
            "percentile_90": 92,
            "std_dev": 11.55015197568389
        }
    }

    
    for age_range, values in age_ranges.items():
        lower, upper = map(float, age_range.split('-'))
        

        if lower <= patient_age < upper:
            median = values['median']
            std_dev = values['std_dev']
            df['necgRate'] = (df['necgRate'] - median) / std_dev
            break


   #################################################################################################################### 
    #z normalise nrespRate and awRr as per Zhiquang Paper
    age_ranges_rr = {
            "0-0.25": {
                "median": 43,
                "percentile_90": 57,
                "std_dev": 8.51063829787234
            },
            "0.25-0.5": {
                "median": 41,
                "percentile_90": 55,
                "std_dev": 8.51063829787234
            },
            "0.5-0.75": {
                "median": 39,
                "percentile_90": 51,
                "std_dev": 7.2948328267477205
            },
            "0.75-1": {
                "median": 37,
                "percentile_90": 50,
                "std_dev": 7.90273556231003
            },
            "1-1.5": {
                "median": 35,
                "percentile_90": 46,
                "std_dev": 6.686930091185411
            },
            "1.5-2": {
                "median": 31,
                "percentile_90": 40,
                "std_dev": 5.47112462006079
            },
            "2-3": {
                "median": 28,
                "percentile_90": 34,
                "std_dev": 3.6474164133738602
            },
            "3-4": {
                "median": 25,
                "percentile_90": 29,
                "std_dev": 2.43161094224924
            },
            "4-6": {
                "median": 23,
                "percentile_90": 27,
                "std_dev": 2.43161094224924
            },
            "6-8": {
                "median": 21,
                "percentile_90": 24,
                "std_dev": 1.8237082066869301
            },
            "8-12": {
                "median": 19,
                "percentile_90": 22,
                "std_dev": 1.8237082066869301
            },
            "12-15": {
                "median": 18,
                "percentile_90": 21,
                "std_dev": 1.8237082066869301
            },
            "15-18": {
                "median": 16,
                "percentile_90": 19,
                "std_dev": 1.8237082066869301
            }
        }
    for age_range, values in age_ranges_rr.items():
        lower, upper = map(float, age_range.split('-'))
        
        if lower <= patient_age < upper:
            median = values['median']
            std_dev = values['std_dev']
            df['nrespRate'] = (df['nrespRate'] - median) / std_dev
            df['nawRr'] = (df['nawRr'] - median) / std_dev
            break

###############################################################################################################################
    # Z-normalize Systolic Blood Pressure (SBP) based on age and sex
    age_ranges_sbp ={
    "M": {
        "0-1": {
            "median": 82,
            "percentile_95": 100,
            "std_dev": 9.183673469387756
        },
        "1-2": {
            "median": 85,
            "percentile_95": 103,
            "std_dev": 9.183673469387756
        },
        "2-3": {
            "median": 88,
            "percentile_95": 106,
            "std_dev": 9.183673469387756
        },
        "3-4": {
            "median": 91,
            "percentile_95": 109,
            "std_dev": 9.183673469387756
        },
        "4-5": {
            "median": 93,
            "percentile_95": 111,
            "std_dev": 9.183673469387756
        },
        "5-6": {
            "median": 95,
            "percentile_95": 112,
            "std_dev": 8.673469387755102
        },
        "6-7": {
            "median": 96,
            "percentile_95": 114,
            "std_dev": 9.183673469387756
        },
        "7-8": {
            "median": 97,
            "percentile_95": 115,
            "std_dev": 9.183673469387756
        },
        "8-9": {
            "median": 99,
            "percentile_95": 117,
            "std_dev": 9.183673469387756
        },
        "9-10": {
            "median": 100,
            "percentile_95": 118,
            "std_dev": 9.183673469387756
        },
        "10-11": {
            "median": 103,
            "percentile_95": 119,
            "std_dev": 8.16326530612245
        },
        "11-12": {
            "median": 104,
            "percentile_95": 121,
            "std_dev": 8.673469387755102
        },
        "12-13": {
            "median": 106,
            "percentile_95": 123,
            "std_dev": 8.673469387755102
        },
        "13-14": {
            "median": 108,
            "percentile_95": 126,
            "std_dev": 9.183673469387756
        },
        "14-15": {
            "median": 111,
            "percentile_95": 128,
            "std_dev": 8.673469387755102
        },
        "15-16": {
            "median": 113,
            "percentile_95": 131,
            "std_dev": 9.183673469387756
        },
        "16-17": {
            "median": 116,
            "percentile_95": 134,
            "std_dev": 9.183673469387756
        },
        "17-18": {
            "median": 118,
            "percentile_95": 136,
            "std_dev": 9.183673469387756
        }
    },
    "F": {
        "0-1": {
            "median": 82,
            "percentile_95": 103,
            "std_dev": 10.714285714285715
        },
        "1-2": {
            "median": 86,
            "percentile_95": 104,
            "std_dev": 9.183673469387756
        },
        "2-3": {
            "median": 88,
            "percentile_95": 105,
            "std_dev": 8.673469387755102
        },
        "3-4": {
            "median": 89,
            "percentile_95": 107,
            "std_dev": 9.183673469387756
        },
        "4-5": {
            "median": 91,
            "percentile_95": 108,
            "std_dev": 8.673469387755102
        },
        "5-6": {
            "median": 93,
            "percentile_95": 110,
            "std_dev": 8.673469387755102
        },
        "6-7": {
            "median": 94,
            "percentile_95": 111,
            "std_dev": 8.673469387755102
        },
        "7-8": {
            "median": 96,
            "percentile_95": 113,
            "std_dev": 8.673469387755102
        },
        "8-9": {
            "median": 98,
            "percentile_95": 115,
            "std_dev": 8.673469387755102
        },
        "9-10": {
            "median": 100,
            "percentile_95": 117,
            "std_dev": 8.673469387755102
        },
        "10-11": {
            "median": 102,
            "percentile_95": 119,
            "std_dev": 8.673469387755102
        },
        "11-12": {
            "median": 103,
            "percentile_95": 121,
            "std_dev": 9.183673469387756
        },
        "12-13": {
            "median": 105,
            "percentile_95": 123,
            "std_dev": 9.183673469387756
        },
        "13-14": {
            "median": 107,
            "percentile_95": 124,
            "std_dev": 8.673469387755102
        },
        "14-15": {
            "median": 109,
            "percentile_95": 126,
            "std_dev": 8.673469387755102
        },
        "15-16": {
            "median": 110,
            "percentile_95": 127,
            "std_dev": 8.673469387755102
        },
        "16-17": {
            "median": 111,
            "percentile_95": 128,
            "std_dev": 8.673469387755102
        },
        "17-18": {
            "median": 111,
            "percentile_95": 129,
            "std_dev": 9.183673469387756
        }
    }
}
    # Retrieve the appropriate age range based on patient age and sex

    # Retrieve the appropriate age range based on patient age and sex
    gender_specific_ranges_sbp = age_ranges_sbp.get(patient_sex, {})

    for age_range, values in gender_specific_ranges_sbp.items():
        lower, upper = map(float, age_range.split('-'))
        if lower <= patient_age < upper:
            median = values['median']
            std_dev = values['std_dev']
            
            # Apply z-score normalization for nartSys and nnbpSys
            df['nartSys'] = (df['nartSys'] - median) / std_dev
            df['nnbpSys'] = (df['nnbpSys'] - median) / std_dev
            break

#######################################################################################################################
    # Z-normalize Diastolic Blood Pressure (DBP) based on age and sex
    age_ranges_dbp ={
    "M": {
        "0-1": {
            "median": 33,
            "percentile_95": 52,
            "std_dev": 9.693877551020408
        },
        "1-2": {
            "median": 37,
            "percentile_95": 56,
            "std_dev": 9.693877551020408
        },
        "2-3": {
            "median": 42,
            "percentile_95": 61,
            "std_dev": 9.693877551020408
        },
        "3-4": {
            "median": 46,
            "percentile_95": 65,
            "std_dev": 9.693877551020408
        },
        "4-5": {
            "median": 50,
            "percentile_95": 69,
            "std_dev": 9.693877551020408
        },
        "5-6": {
            "median": 53,
            "percentile_95": 72,
            "std_dev": 9.693877551020408
        },
        "6-7": {
            "median": 55,
            "percentile_95": 74,
            "std_dev": 9.693877551020408
        },
        "7-8": {
            "median": 57,
            "percentile_95": 76,
            "std_dev": 9.693877551020408
        },
        "8-9": {
            "median": 59,
            "percentile_95": 78,
            "std_dev": 9.693877551020408
        },
        "9-10": {
            "median": 60,
            "percentile_95": 79,
            "std_dev": 9.693877551020408
        },
        "10-11": {
            "median": 61,
            "percentile_95": 80,
            "std_dev": 9.693877551020408
        },
        "11-12": {
            "median": 61,
            "percentile_95": 80,
            "std_dev": 9.693877551020408
        },
        "12-13": {
            "median": 62,
            "percentile_95": 81,
            "std_dev": 9.693877551020408
        },
        "13-14": {
            "median": 62,
            "percentile_95": 81,
            "std_dev": 9.693877551020408
        },
        "14-15": {
            "median": 63,
            "percentile_95": 82,
            "std_dev": 9.693877551020408
        },
        "15-16": {
            "median": 64,
            "percentile_95": 83,
            "std_dev": 9.693877551020408
        },
        "16-17": {
            "median": 65,
            "percentile_95": 84,
            "std_dev": 9.693877551020408
        },
        "17-18": {
            "median": 67,
            "percentile_95": 87,
            "std_dev": 10.204081632653061
        }
    },
    "F": {
        "0-1": {
            "median": 43,
            "percentile_95": 53,
            "std_dev": 5.1020408163265305
        },
        "1-2": {
            "median": 40,
            "percentile_95": 58,
            "std_dev": 9.183673469387756
        },
        "2-3": {
            "median": 45,
            "percentile_95": 63,
            "std_dev": 9.183673469387756
        },
        "3-4": {
            "median": 49,
            "percentile_95": 67,
            "std_dev": 9.183673469387756
        },
        "4-5": {
            "median": 52,
            "percentile_95": 70,
            "std_dev": 9.183673469387756
        },
        "5-6": {
            "median": 54,
            "percentile_95": 72,
            "std_dev": 9.183673469387756
        },
        "6-7": {
            "median": 56,
            "percentile_95": 74,
            "std_dev": 9.183673469387756
        },
        "7-8": {
            "median": 57,
            "percentile_95": 75,
            "std_dev": 9.183673469387756
        },
        "8-9": {
            "median": 58,
            "percentile_95": 76,
            "std_dev": 9.183673469387756
        },
        "9-10": {
            "median": 59,
            "percentile_95": 77,
            "std_dev": 9.183673469387756
        },
        "10-11": {
            "median": 60,
            "percentile_95": 78,
            "std_dev": 9.183673469387756
        },
        "11-12": {
            "median": 61,
            "percentile_95": 79,
            "std_dev": 9.183673469387756
        },
        "12-13": {
            "median": 62,
            "percentile_95": 80,
            "std_dev": 9.183673469387756
        },
        "13-14": {
            "median": 63,
            "percentile_95": 81,
            "std_dev": 9.183673469387756
        },
        "14-15": {
            "median": 64,
            "percentile_95": 82,
            "std_dev": 9.183673469387756
        },
        "15-16": {
            "median": 65,
            "percentile_95": 83,
            "std_dev": 9.183673469387756
        },
        "16-17": {
            "median": 66,
            "percentile_95": 84,
            "std_dev": 9.183673469387756
        },
        "17-18": {
            "median": 66,
            "percentile_95": 84,
            "std_dev": 9.183673469387756
        }
    }
}
    # Retrieve the appropriate age range based on patient age and sex for DBP
    gender_specific_ranges_dbp = age_ranges_dbp.get(patient_sex, {})
        
    for age_range, values in gender_specific_ranges_dbp.items():
        lower, upper = map(float, age_range.split('-'))
        if lower <= patient_age < upper:
            median = values['median']
            std_dev = values['std_dev']
            
            # Apply z-score normalization for nartSys and nnbpSys
            df['nartDia'] = (df['nartDia'] - median) / std_dev
            df['nnbpDia'] = (df['nnbpDia'] - median) / std_dev
            break
            
###################################################################################################################
    #######################################################################################################################
    # Z-normalize Mean Blood Pressure (MBP) based on age and sex
    age_ranges_mbp = {
    "M": {
        "0-1": {
            "median": 50,
            "percentile_95": 68,
            "std_dev": 9.183673469387756
        },
        "1-2": {
            "median": 53,
            "percentile_95": 72,
            "std_dev": 9.693877551020408
        },
        "2-3": {
            "median": 57,
            "percentile_95": 76,
            "std_dev": 9.693877551020408
        },
        "3-4": {
            "median": 61,
            "percentile_95": 80,
            "std_dev": 9.693877551020408
        },
        "4-5": {
            "median": 64,
            "percentile_95": 83,
            "std_dev": 9.693877551020408
        },
        "5-6": {
            "median": 67,
            "percentile_95": 85,
            "std_dev": 9.183673469387756
        },
        "6-7": {
            "median": 69,
            "percentile_95": 87,
            "std_dev": 9.183673469387756
        },
        "7-8": {
            "median": 70,
            "percentile_95": 89,
            "std_dev": 9.693877551020408
        },
        "8-9": {
            "median": 72,
            "percentile_95": 91,
            "std_dev": 9.693877551020408
        },
        "9-10": {
            "median": 73,
            "percentile_95": 92,
            "std_dev": 9.693877551020408
        },
        "10-11": {
            "median": 75,
            "percentile_95": 93,
            "std_dev": 9.183673469387756
        },
        "11-12": {
            "median": 75,
            "percentile_95": 94,
            "std_dev": 9.693877551020408
        },
        "12-13": {
            "median": 77,
            "percentile_95": 95,
            "std_dev": 9.183673469387756
        },
        "13-14": {
            "median": 77,
            "percentile_95": 96,
            "std_dev": 9.693877551020408
        },
        "14-15": {
            "median": 79,
            "percentile_95": 97,
            "std_dev": 9.183673469387756
        },
        "15-16": {
            "median": 80,
            "percentile_95": 99,
            "std_dev": 9.693877551020408
        },
        "16-17": {
            "median": 82,
            "percentile_95": 101,
            "std_dev": 9.693877551020408
        },
        "17-18": {
            "median": 84,
            "percentile_95": 103,
            "std_dev": 9.693877551020408
        }
    },
    "F": {
        "0-1": {
            "median": 51,
            "percentile_95": 69,
            "std_dev": 9.183673469387756
        },
        "1-2": {
            "median": 55,
            "percentile_95": 73,
            "std_dev": 9.183673469387756
        },
        "2-3": {
            "median": 59,
            "percentile_95": 77,
            "std_dev": 9.183673469387756
        },
        "3-4": {
            "median": 62,
            "percentile_95": 80,
            "std_dev": 9.183673469387756
        },
        "4-5": {
            "median": 65,
            "percentile_95": 83,
            "std_dev": 9.183673469387756
        },
        "5-6": {
            "median": 67,
            "percentile_95": 85,
            "std_dev": 9.183673469387756
        },
        "6-7": {
            "median": 69,
            "percentile_95": 86,
            "std_dev": 8.673469387755102
        },
        "7-8": {
            "median": 70,
            "percentile_95": 88,
            "std_dev": 9.183673469387756
        },
        "8-9": {
            "median": 71,
            "percentile_95": 89,
            "std_dev": 9.183673469387756
        },
        "9-10": {
            "median": 73,
            "percentile_95": 90,
            "std_dev": 8.673469387755102
        },
        "10-11": {
            "median": 74,
            "percentile_95": 92,
            "std_dev": 9.183673469387756
        },
        "11-12": {
            "median": 75,
            "percentile_95": 93,
            "std_dev": 9.183673469387756
        },
        "12-13": {
            "median": 76,
            "percentile_95": 94,
            "std_dev": 9.183673469387756
        },
        "13-14": {
            "median": 78,
            "percentile_95": 95,
            "std_dev": 8.673469387755102
        },
        "14-15": {
            "median": 79,
            "percentile_95": 97,
            "std_dev": 9.183673469387756
        },
        "15-16": {
            "median": 80,
            "percentile_95": 98,
            "std_dev": 9.183673469387756
        },
        "16-17": {
            "median": 81,
            "percentile_95": 99,
            "std_dev": 9.183673469387756
        },
        "17-18": {
            "median": 81,
            "percentile_95": 99,
            "std_dev": 9.183673469387756
        }
    }
}

    # Retrieve the appropriate age range based on patient age and sex for MBP
    gender_specific_ranges_mbp = age_ranges_mbp.get(patient_sex, {})
    for age_range, values in gender_specific_ranges_mbp.items():
        lower, upper = map(float, age_range.split('-'))
        if lower <= patient_age < upper:
            median = values['median']
            std_dev = values['std_dev']
            
            # Apply z-score normalization for nartSys and nnbpSys
            df['nartMean'] = (df['nartMean'] - median) / std_dev
            df['nnbpMean'] = (df['nnbpMean'] - median) / std_dev
            break
##################################################################################################################
#min max scale sats using normal range (max 100, min 94)

    df['nPleth'] = ((df['nPleth']-97)/6)  #normal values should be between 94 and 100 - these scaled to 0-1. 
    #lower values will be negative e.g. stas 80 will scale to -2.3
            
    

####################################################################################################################
#min max scale etco2 max using normal range (max 6, min 4.5), leave nCooMin unscaled as this naturally falls low.

    df['nCoo'] = ((df['nCoo']-5.25)/1.5)
    df['nacooMin'] = ((df['nacooMin']-5.25)/1.5)
    
    
##################################################################################################################
#min max scale sats using temp columns using normal range (max 37.5, min 36.0)

    temp_columns = ['ntempEs', 'ntempSk', 'ntempCo', 'nTemp']
    df[temp_columns] = (df[temp_columns] - 36.75) / 1.5

    #############################################################################
    final_column_order = [                     
                        'nacooMin',
                        'nCoo',
                        'ntempEs',
                        'ntempSk',
                        'ntempCo',
                        'nTemp',
                        'nawRr',
                        'nrespRate',
                        'nartMean',
                        'nartDia',
                        'nartSys',
                        'nnbpMean',
                        'nnbpDia',
                        'nnbpSys',
                        'nPleth',
                        'necgRate'
]
    df = df[final_column_order]
    
    return df

def average_over_1_min(df):
    minute_df = df.copy()
    minute_df['devTime'] = pd.to_timedelta(minute_df['devTime'])
    minute_df.set_index('devTime', inplace=True)
    minute_df = minute_df.resample('1T').mean().round(1)  # '1T' is shorthand for 1 minute reamples ofver next min so e.g. 10m25s and 10m50s readings are resampled to 10m
    # Assuming minute_df is already created and contains 'devTime' as a timedelta
    minute_df.reset_index(inplace=True)


        # Create a new column that represents time in minutes starting from 0
    minute_df['minutes_since_start'] = (minute_df['devTime'] - minute_df['devTime'].min()).dt.total_seconds() / 60

    return minute_df


def calculate_bollinger_bands(df, span=15, window=15, resp_factor=1, cardiac_factor=1.28, average_type='ema'):
    bollinger_bands_df = pd.DataFrame(index=df.index)
    bollinger_bands_df['devTime'] = df['devTime']
    
    # Identify columns that start with 'n'
    n_columns = [col for col in df.columns if col.startswith('n')]

    # Define the column categories
    resp_cols = ['nPleth', 'nCoo', 'nawRr', 'nrespRate']
    cardiac_cols = ['nartMean', 'nnbpMean', 'necgRate']

    # Loop over each column and calculate Bollinger Bands
    for col in n_columns:
        # If the column doesn't fall into either category, skip it
        if col not in resp_cols and col not in cardiac_cols:
            continue
        
        # Filter out NaNs to avoid rolling calculations on NaNs
        valid_data = df[col].dropna()
        
        # Calculate the rolling mean and std dev depending on the average_type
        if average_type == 'sma':
            mb = valid_data.rolling(window=window, min_periods=1).mean()
            std_dev = valid_data.rolling(window=window, min_periods=1).std()
        elif average_type == 'ema': 
            mb = valid_data.ewm(span=span, adjust=False).mean()
            std_dev = valid_data.ewm(span=span, adjust=False).std()

        # Determine factor based on column type
        if col in resp_cols:
            factor = resp_factor
        else:  # col in cardiac_cols
            factor = cardiac_factor

        # Calculate upper and lower bands
        ub = mb + (std_dev * factor)
        lb = mb - (std_dev * factor)

        # Adds some buffer to prevent oversensitivity
        if col in ['nawRr','nrespRate', 'nartMean', 'nnbpMean','nCoo', 'necgRate']:
            ub = np.maximum(1.05 * mb, ub)
            lb = np.minimum(0.95 * mb, lb)
        elif col in ['nPleth']:
            ub = np.maximum(1.025 * mb, ub)
            lb = np.minimum(0.975 * mb, lb)

        # Add the bands to the DataFrame, aligning with the original index
        bollinger_bands_df[f'{col}_UB'] = ub.reindex(df.index)
        bollinger_bands_df[f'{col}_LB'] = lb.reindex(df.index)

    # Concatenate the original DataFrame with the Bollinger Bands DataFrame
    bollinger_bands_df = pd.concat([df, bollinger_bands_df.drop(columns='devTime')], axis=1)

    return bollinger_bands_df



def identify_resp_deteriorations(minute_df):
    
    # Identify rows where deterioration conditions are met
    # Assuming you have calculated Bollinger Bands and added them to minute_df for each relevant column
    # Modify this to use Upper Band (UB) and Lower Band (LB)
    sats_threshold = np.minimum(94, minute_df['nPleth_LB']) # creates a list of sats thresholds and each one is used in turn


    deteriorations_df = minute_df[
        (minute_df['nPleth'] < sats_threshold) &  # Condition for nPleth based on its Lower Band
        (
            (minute_df['nCoo'] > minute_df['nCoo_UB']) |  # nCoo greater than Upper Band
            (minute_df['nCoo'] < minute_df['nCoo_LB']) |  # nCoo less than Lower Band
            (minute_df['nrespRate'] > minute_df['nrespRate_UB']) |  # nnResp greater than Upper Band
            (minute_df['nrespRate'] < minute_df['nrespRate_LB']) |  # nnResp less than Lower Band
            (minute_df['nawRr'] > minute_df['nawRr_UB']) |  # nacooMin greater than Upper Band
            (minute_df['nawRr'] < minute_df['nawRr_LB'])  # nacooMin less than Lower Band
        )
    ].copy()  # Ensure a copy is made

    # Function to check if at least one of the columns has no NaN values in the preceding and following 5-minute window
    def check_time_window(row, df):
        current_time = row['devTime']

        
        # Define the time window: 5 minutes before and after
        time_window_start = current_time - pd.Timedelta(minutes=5)
        time_window_end = current_time + pd.Timedelta(minutes=5)
        
        # Extract data for the same patient within the time window
        window_df = df[(df['devTime'] >= time_window_start) & 
                        (df['devTime'] <= time_window_end)]  # Ensure we only get data for the same patient
        
        # Columns to check for NaN values
        columns_to_check = ['nPleth', 'nCoo', 'nawRr', 'nrespRate']
        
        # Check if any column has at least 10 non-NaN values
        for column in columns_to_check:
            non_nan_count = window_df[column].notna().sum()  # Count non-NaN values
            if non_nan_count >= 11:  # Check if there are at least 10 valid data points
                return True  # At least one column has 10 complete minutes of data

        return False  # No column has 11 complete minutes of data

    # Apply the check for each row in deteriorations_df
    deteriorations_df.loc[:, 'has_full_10min_window'] = deteriorations_df.apply(lambda row: check_time_window(row, minute_df), axis=1)

    # Filter to only include rows that have data in the 10-minute window
    valid_deteriorations_df = deteriorations_df[deteriorations_df['has_full_10min_window']]

    return valid_deteriorations_df





def identify_cardiac_deteriorations(minute_df):
    
    bp_columns = ['nartMean', 'nnbpMean']
    
    # Create lists to hold conditions
    above_bands_conditions = []
    below_bands_conditions = []
    
    
    
    # Iterate through each BP column and its corresponding UB and LB
    for col in bp_columns:
        # Handle NaN values by using fillna(False)
        above_bands_conditions.append((minute_df[col] > minute_df[col + '_UB']).fillna(False))
        below_bands_conditions.append((minute_df[col] < minute_df[col + '_LB']).fillna(False))
    
    # Sum the True values for each row to check how many BP readings are above or below their respective bands
    above_bands = sum(above_bands_conditions)
    below_bands = sum(below_bands_conditions)


    # Condition 1: At least 2 readings above or at least 2 readings below
    bp_condition = (above_bands >= 2) | (below_bands >= 2)
    
    # Condition 2: Significant change in heart rate (HR), handle NaN by skipping
    heart_rate_condition = (
        (minute_df['necgRate'] > minute_df['necgRate_UB']) | 
        (minute_df['necgRate'] < minute_df['necgRate_LB'])
    ).fillna(False)
    
    # Condition 3: Single BP reading moves above or below the bands, handle NaN by skipping
    bp_single_change_condition = (
        (minute_df['nartMean'] > minute_df['nartMean_UB']) | (minute_df['nartMean'] < minute_df['nartMean_LB']) |
        (minute_df['nnbpMean'] > minute_df['nnbpMean_UB']) | (minute_df['nnbpMean'] < minute_df['nnbpMean_LB'])
    ).fillna(False)
    
    # Final condition: 
    # 1. Either BP condition is met
    # 2. Or both HR condition and BP single change condition are met
    deterioration_condition = bp_condition | (heart_rate_condition & bp_single_change_condition)

    # Filter rows where deterioration conditions are met
    deteriorations_df = minute_df[deterioration_condition].copy()

     # Function to check if at least one of the columns has no NaN values in the preceding and following 5-minute window
    def check_time_window(row, df):
        current_time = row['devTime']
   
        # Define the time window: 5 minutes before and after
        time_window_start = current_time - pd.Timedelta(minutes=5)
        time_window_end = current_time + pd.Timedelta(minutes=5)
        
        # Extract data for the same patient within the time window
        window_df = df[(df['devTime'] >= time_window_start) & 
                        (df['devTime'] <= time_window_end)] 
        
        # Columns to check for NaN values
        columns_to_check = ['necgRate', 'nartMean', 'nnbpMean']
        
        # Check if any column has at least 10 non-NaN values
        for column in columns_to_check:
            non_nan_count = window_df[column].notna().sum()  # Count non-NaN values
            if non_nan_count >= 11:  # Check if there are at least 10 valid data points
                return True  # At least one column has 10 complete minutes of data

        return False  # No column has 11 complete minutes of data

    # Apply the check for each row in deteriorations_df
    deteriorations_df.loc[:, 'has_full_10min_window'] = deteriorations_df.apply(lambda row: check_time_window(row, minute_df), axis=1)

    # Filter to only include rows that have data in the 10-minute window
    valid_deteriorations_df = deteriorations_df[deteriorations_df['has_full_10min_window']]

    return valid_deteriorations_df




def generate_deterioration_labels(resp_df,cardiac_df, window_end_time, prediction_window_length=15):

    if not resp_df.empty:
        resp_df = resp_df[
            ((resp_df['minutes_since_start']>=window_end_time))&  ## can be greater or equal to here because e.g. 15 min window ends with last timepoint at 14:59 so no data leakage 
            (resp_df['minutes_since_start']< (window_end_time+prediction_window_length))] #like wise this should be < not <= say because up to min 29 includes data up to 29:59 so 15 minute window. if was <= would include say minute 30 which incorporates data up to muinute 30:59

    if not cardiac_df.empty:
        cardiac_df = cardiac_df[
            ((cardiac_df['minutes_since_start']>=window_end_time))&  ## can be greater or equal to here because e.g. 15 min window ends with last timepoint at 14:59 so no data leakage 
            (cardiac_df['minutes_since_start']< (window_end_time+prediction_window_length))]#like wise this should be < not <= say because up to min 29 includes data up to 29:59 so 15 minute window. if was <= would include say minute 30 which incorporates data up to muinute 30:59

  # Return binary list: [0 or 1 for resp_df, 0 or 1 for cardiac_df]
    return [
        1 if not resp_df.empty else 0,  # 1 if resp_df is not empty, 0 if empty
        1 if not cardiac_df.empty else 0  # 1 if cardiac_df is not empty, 0 if empty
        ]




def load_data(file_name):
    directory = r'/home/workspace/files/'    
    file_path = os.path.join(directory, file_name)
    
    # Step 1: Read the CSV file without enforcing dtype to avoid errors
    df = pd.read_csv(file_path, low_memory=False)
    
    


    df = filter_and_clean_columns(df)
    df = process_time_intervals(df)
    df = remove_physiologically_implausible_values(df, min_hr = 30, max_hr = 300,max_etco2 = 15)
    df = process_bp_columns(df)
    #df = add_shock_index_column(df)
    
    scaled_values = scale_values(df.copy())

    
    df = average_over_1_min(df)    
    df = calculate_bollinger_bands(df,span = 15, resp_factor = 1, cardiac_factor=1.28, average_type = 'ema')
    resp_df = identify_resp_deteriorations(df)
    cardiac_df = identify_cardiac_deteriorations(df)
    

        

    # Return list with scaled values DataFrame and label pair
    return [scaled_values, resp_df, cardiac_df]

    
    




#########################################################################################################################

def get_num_minutes(filename):
    directory = r'/home/workspace/files/'    
    file_path = os.path.join(directory, filename)
    
    # Specify columns to read for efficiency
    use_cols = ['devTime', 'nPleth', 'necgRate', 'nacooMin', 'nCoo', 
                'ntempEs', 'ntempSk', 'ntempCo', 'nTemp', 
                'nawRr', 'nrespRate', 'nartMean', 'nartDia', 
                'nartSys', 'nabpMean', 'nabpDia', 'nabpSys', 
                'nnbpMean', 'nnbpDia', 'nnbpSys']
    
    # Step 1: Load only necessary columns, with devTime parsed as datetime
    df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)
    
    # Step 2: Drop completely empty rows based on specified columns
    df[use_cols[1:]] = df[use_cols[1:]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=use_cols[1:], how='all')
    
    # Step 3: Convert 'devTime' to datetime
    df['devTime'] = pd.to_datetime(df['devTime'], unit='s')
    
    # Step 4: Adjust 'devTime' to be relative to the minimum timestamp
    min_time = df['devTime'].min()
    df['devTime'] = df['devTime'] - min_time
    
    df.set_index('devTime', inplace=True)
    df = df.resample('1T').mean().round(1)  # '1T' is shorthand for 1 minute reamples ofver next min so e.g. 10m25s and 10m50s readings are resampled to 10m
    # Assuming minute_df is already created and contains 'devTime' as a timedelta

    num_rows = df.shape[0]-1 #to account for the fact the first minute is labbeled as zero'th minute

    return num_rows
    
    
#########################################################################################################################










