import sys


from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import string
import pandas as pd
import numpy as np
from generating_datasets_for_torch import *
from load_static_data import *


class PatientDataset(Dataset):
    
    def __init__(self, patient_list, min_window_min=15, step_min=1, max_window_min=60, prediction_window_length=15):
        
        # Assuming patient_list is a list of file names like ['001_patient.csv', '002_patient.csv', ...]    
        #  create a dictionary mapping cats_id to file_path
        self.patient_list = patient_list
        self.files = {}
        for file_name in self.patient_list:
            cats_id = int(file_name.split('_')[0])  # Extract the integer ID from the file name
            self.files[cats_id] = file_name  # Map cats_id to file_path
        
        self.min_window_min = min_window_min
        self.step_min = step_min
        self.max_window_min = max_window_min
        self.prediction_window_length = prediction_window_length
        self.max_window_rows = self.max_window_min * 60  # Assuming one row per second
        
        self.samples = []  # List of tuples (cats_id, start_min, end_min)
        self.cache = {}    # Cache to store loaded DataFrames

        # Wrap tqdm around the outer loop to monitor progress over each patient file
        
        
        #To ensure label window accurately represents full 15 minute window need to stop sliding windows 
        #going all the way to the end and leaving 15 minute period  (or other prediction_window_leght e.g 10mins)
        #at end for label period to occur. 
        #otherwise label window is not truly representative
        
        self.samples = []
        for cats_id, file_name in tqdm(self.files.items(), desc="Generating samples"):
            total_minutes = get_num_minutes(file_name)
            latest_end_min = total_minutes - self.prediction_window_length

            patient_samples = self.generate_windows(cats_id, latest_end_min)
            self.samples.extend(patient_samples)


               
        self.static_data = load_static_data(self.patient_list)
        self.static_data.set_index('cats_id', inplace=True)


    ################note each window starts and ends on the minute i.e. minute starts at 00:00 and would end e.g. at 00:59s. 
#e.g. first time window engs at 14:59 so fair game to predict events from minute 15 as has not yet seen this data
#this is CRITICAL for preventing time leakage i.e. the system from already having seen the event it will then label in training
    def generate_windows(self, cats_id, latest_end_min):
        """
        Generates two types of windows:
        1. Growing windows: Starting at 0, increasing in size up to max_window_min
        2. Sliding windows: Fixed max_window_min size, sliding forward

        Example with min_window_min=15, step_min=15, max_window_min=60:
        Growing windows:  (0,15), (0,30), (0,45), (0,60)
        Sliding windows:  (15,75), (30,90), ... until latest_end_min

        Args:
            cats_id (int): The patient ID
            latest_end_min (int): The latest possible end minute (file_length - prediction_window)
        """


        # Step 1: Growing windows starting at 0
        # Generate windows that grow in size: 0-15, 0-30, 0-45, 0-60
        growing_windows = np.arange(
            self.min_window_min,  # e.g., 15
            min(self.max_window_min + 1, latest_end_min + 1),  # e.g., 61 or less if file is shorter
            self.step_min  # e.g., 15
        )
        # Filter windows that would exceed the file length minus prediction window
        valid_growing = growing_windows[growing_windows <= latest_end_min]
        growing_samples = [(cats_id, 0, int(window)) for window in valid_growing]

        # Step 2: Sliding windows of max_window_min size
        # Only generate sliding windows if we have enough data
        if latest_end_min > self.max_window_min:
            # Start points: 15, 30, 45, ... until (latest_end_min - max_window_min)
            start_minutes = np.arange(
                self.min_window_min,  # e.g., 15
                latest_end_min - self.max_window_min + 1,  # Adjust to ensure last window fits
                self.step_min # e.g., 15 min step
            )
            # For each start point, the end point is start + max_window_min
            end_minutes = start_minutes + self.max_window_min  # e.g., start + 60

            # Create samples for all valid sliding windows
            sliding_samples = list(zip(
                [cats_id] * len(start_minutes),
                start_minutes.astype(int),
                end_minutes.astype(int)
            ))
        else:
            sliding_samples = []

        return growing_samples + sliding_samples
#####################################################################################################################

    def __len__(self):
        # Return the number of samples (not files)
        return len(self.samples)
    ##################################################################################################################
    def __getitem__(self, idx):
        # Get the sample (cats_id, start_min, end_min)
        cats_id, start_min, end_min = self.samples[idx]

        # Load the data for the patient if not cached
        if cats_id not in self.cache:
            file_path = self.files[cats_id]
            data,resp_df,cardiac_df = load_data(file_path)  # Assuming this function loads your CSV into a DataFrame
            self.cache[cats_id] = (data,resp_df,cardiac_df)

        # Extract the window data from start_min to end_min (convert minutes to rows)
        data = self.cache[cats_id][0]
        start_row = start_min * 60
        end_row = end_min * 60
        
        
        ############### Generate labels
        resp_df = self.cache[cats_id][1]
        cardiac_df = self.cache[cats_id][2]
        labels = generate_deterioration_labels(resp_df, cardiac_df, end_min, self.prediction_window_length)
        label_tensor = torch.tensor(labels, dtype=torch.float32)       
        
        #######################################        
#         # add 5 minutely averages over up to previous 2 hours of data before the 15 minute block of 1 second data. 
        summary_df = data.iloc[0:start_row]        

        # Define constants
        rows_per_interval = 300  # seconds in 5 minutes
        max_seconds = 7200       # 2 hours in seconds
        num_intervals = max_seconds // rows_per_interval  # Expected number of 5-min intervals (should be 24)

        # 1. Truncate summary_df to the most recent 2 hours if necessary
        if len(summary_df) > max_seconds:
            summary_df = summary_df.iloc[-max_seconds:]

        # 2. Group data into 5-minute intervals and compute the mean for each interval
        # Create a grouping index for each 300-row block
        grouping = np.arange(len(summary_df)) // rows_per_interval
        five_min_avg = summary_df.groupby(grouping).mean()

        # 3. Pad with NaNs at the top if fewer than 24 intervals
        current_intervals = len(five_min_avg)
        if current_intervals < num_intervals:
            missing_rows = num_intervals - current_intervals
            # Create a padding DataFrame with NaN values
            pad = pd.DataFrame(np.nan, index=range(missing_rows), columns=five_min_avg.columns)
            # Concatenate the NaN rows on top of the computed averages
            five_min_avg = pd.concat([pad, five_min_avg], ignore_index=True)

        # Optionally, reset the index if you want it from 0 to 23
    
    
        # Generate the first 24 letters of the alphabet
        alphabetical_index = list(string.ascii_uppercase[:24])

        # Assign the new alphabetical index to the DataFrame
        five_min_avg.index = alphabetical_index

        # five_min_avg now contains 24 rows representing 5-minute averages, padded with NaNs if needed       
    
                    ##########
        window_data = data.iloc[start_row:end_row]
        #now we add five minute avg before window data
        window_data=pd.concat([five_min_avg,window_data])       
    
        
        #####################################


        # Ensure 'cumulative_resp' column exists and is initialized to 0

        window_data = window_data.copy()  # Make sure you're working on a real copy
        window_data['cumulative_resp'] = 0


            
        if not resp_df.empty:
            resp_df['seconds_since_start'] = pd.to_timedelta(resp_df['devTime']).dt.total_seconds()
            resp_df = resp_df[resp_df['seconds_since_start'] < end_row]

            # Get all index values in window_data where the index matches any value in resp_df['seconds_since_start']
            matching_indices = window_data.index.intersection(resp_df['seconds_since_start'])

            # Use vectorized operations to set matching rows and the next 59 rows to 1
            for idx in matching_indices:
                # Set the current index and the next 59 indices to 1 (within bounds)
                window_data.loc[idx:idx + 59, 'cumulative_resp'] = 1
                
            for i, row in resp_df.iterrows():
                # How many minutes before the start of the main window?
                if row['seconds_since_start'] < start_row:
                    sec_before = start_row - row['seconds_since_start']
                    min_before = sec_before / 60.0  # e.g. 118.0 means 118 minutes before

                    # Identify which 5-minute block covers min_before.
                    # "A" covers [120..115), "B" covers [115..110), etc. up to "X" covers [5..0).
                    block_index = None
                    for block_i in range(24):
                        # For block_i=0 => 'A': [120..115)
                        # For block_i=1 => 'B': [115..110)
                        # ...
                        lower_bound = 120 - 5 * (block_i + 1)  # inclusive
                        upper_bound = 120 - 5 * block_i        # exclusive
                        if lower_bound <= min_before < upper_bound:
                            block_index = block_i
                            break

                    if block_index is not None:
                        letter = alphabetical_index[block_index]
                        # Set cumulative_resp=1 in that particular row
                        window_data.loc[letter, 'cumulative_resp'] = 1

#####################################################################################
        ##now cardiac_DF labels
        window_data = window_data.copy()  # Make sure you're working on a real copy
        window_data['cumulative_cardiac'] = 0

        if not cardiac_df.empty:
            cardiac_df['seconds_since_start'] = pd.to_timedelta(cardiac_df['devTime']).dt.total_seconds()
            cardiac_df = cardiac_df[cardiac_df['seconds_since_start'] < end_row]

            # Get all index values in window_data where the index matches any value in resp_df['seconds_since_start']
            matching_indices = window_data.index.intersection(cardiac_df['seconds_since_start'])

            # Use vectorized operations to set matching rows and the next 59 rows to 1
            for idx in matching_indices:
                # Set the current index and the next 59 indices to 1 (within bounds)
                window_data.loc[idx:idx + 59, 'cumulative_cardiac'] = 1
                
            for i, row in cardiac_df.iterrows():
                # How many minutes before the start of the main window?
                if row['seconds_since_start'] < start_row:
                    sec_before = start_row - row['seconds_since_start']
                    min_before = sec_before / 60.0  # e.g. 118.0 means 118 minutes before

                    # Identify which 5-minute block covers min_before.
                    # "A" covers [120..115), "B" covers [115..110), etc. up to "X" covers [5..0).
                    block_index = None
                    for block_i in range(24):
                        # For block_i=0 => 'A': [120..115)
                        # For block_i=1 => 'B': [115..110)
                        # ...
                        lower_bound = 120 - 5 * (block_i + 1)  # inclusive
                        upper_bound = 120 - 5 * block_i        # exclusive
                        if lower_bound <= min_before < upper_bound:
                            block_index = block_i
                            break

                    if block_index is not None:
                        letter = alphabetical_index[block_index]
                        # Set cumulative_resp=1 in that particular row
                        window_data.loc[letter, 'cumulative_cardiac'] = 1

                # Print the updated DataFrame to verify


        #######################################
        

        # Generate the missingness mask
        mask = ~window_data.isna()
        mask_np = mask.to_numpy(dtype=float)
        missingness_mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

        # Handle NaN values in features by ffill then imputing still missing values (from beginning) to 0 - this is neutral and better than -1 as 
        #we have z normalised. We will pass missingness mask anyway so these should have minimal attention paid to them
        features = window_data.ffill()
        features = window_data.fillna(0)
        features  = features.to_numpy(dtype=float)
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Determine the number of rows in the current window
        current_window_rows = window_data.shape[0]

        # Padding if needed
        if current_window_rows < self.max_window_rows:
            padding_length = self.max_window_rows - current_window_rows
            pad_features = torch.zeros((padding_length, feature_tensor.shape[1]), dtype=torch.float32)
            feature_tensor = torch.cat([feature_tensor, pad_features], dim=0)

            pad_mask = torch.zeros((padding_length, missingness_mask_tensor.shape[1]), dtype=torch.float32)
            missingness_mask_tensor = torch.cat([missingness_mask_tensor, pad_mask], dim=0)

            padding_mask = torch.cat([torch.ones(current_window_rows, dtype=torch.float32),
                                      torch.zeros(padding_length, dtype=torch.float32)], dim=0)
        else:
            padding_mask = torch.ones(self.max_window_rows, dtype=torch.float32)

       
    
        # Load static data for the current window
        static_row = self.static_data.loc[cats_id]
        # Directly pop the 'primary_diagnosis_embedding' column and flatten it
        list_features = static_row.pop('primary_diagnosis_embedding')  # This is already a list, so no need to extend

        # Convert remaining scalar features to a tensor
        scalar_tensor = torch.tensor(static_row.to_numpy(dtype=float), dtype=torch.float32)
        # Convert list features to a tensor and concatenate with scalar features
        list_tensor = torch.tensor(list_features, dtype=torch.float32)
        static_tensor = torch.cat([scalar_tensor, list_tensor], dim=0)




        # Return all tensors
        return (
            feature_tensor,
            missingness_mask_tensor,
            padding_mask,
            static_tensor,
            scalar_tensor,
            list_tensor,
            label_tensor,
            torch.tensor(cats_id),
            torch.tensor(start_min),
            torch.tensor(end_min),
            torch.tensor(self.prediction_window_length)
        )
    


