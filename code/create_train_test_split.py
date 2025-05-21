import pandas as pd
import numpy as np
import os
# Load the data

def create_train_test_split(patient_list, seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Load the patient data
    df = pd.read_csv(patient_list)
    
    # Group by patient to summarize the deterioration episodes
    patient_groups = df.groupby('cats_id')
    patients_summary = patient_groups.agg(
        any_resp_deterioration=('resp_deterioration', 'max'),
        any_cardiac_deterioration=('cardiac_deterioration', 'max')
    )

    # Define categories based on the type of deterioration
    only_resp_patients = patients_summary[(patients_summary['any_resp_deterioration'] == 1) & (patients_summary['any_cardiac_deterioration'] == 0)].index
    only_cardiac_patients = patients_summary[(patients_summary['any_resp_deterioration'] == 0) & (patients_summary['any_cardiac_deterioration'] == 1)].index
    both_patients = patients_summary[(patients_summary['any_resp_deterioration'] == 1) & (patients_summary['any_cardiac_deterioration'] == 1)].index
    neither_patients = patients_summary[(patients_summary['any_resp_deterioration'] == 0) & (patients_summary['any_cardiac_deterioration'] == 0)].index

    # Helper function to split indices into train, val, and test
    def stratified_split(indices):
        np.random.shuffle(indices)  # Random shuffle for random split
        train_end = int(0.8 * len(indices))
        val_end = train_end + int(0.1 * len(indices))
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    # Split each category into train, val, and test
    train_indices, val_indices, test_indices = [], [], []

    for patient_category in [only_resp_patients, only_cardiac_patients, both_patients, neither_patients]:
        train, val, test = stratified_split(list(patient_category))
        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)

    # Create DataFrames for each set by filtering the original DataFrame based on the indices
    train_df = df[df['cats_id'].isin(train_indices)]
    val_df = df[df['cats_id'].isin(val_indices)]
    test_df = df[df['cats_id'].isin(test_indices)]
    


    
#######################################################################################################################
    ##check and output proportions
    def calculate_proportion(df, name):
        # Group by patient and check if any deterioration occurred
        patient_groups = df.groupby('cats_id').agg(
            any_resp_deterioration=('resp_deterioration', 'max'),
            any_cardiac_deterioration=('cardiac_deterioration', 'max')
        )

        # Calculate counts for each patient category
        only_resp_count = ((patient_groups['any_resp_deterioration'] == 1) & 
                           (patient_groups['any_cardiac_deterioration'] == 0)).sum()
        only_cardiac_count = ((patient_groups['any_resp_deterioration'] == 0) & 
                              (patient_groups['any_cardiac_deterioration'] == 1)).sum()
        both_count = ((patient_groups['any_resp_deterioration'] == 1) & 
                      (patient_groups['any_cardiac_deterioration'] == 1)).sum()
        neither_count = ((patient_groups['any_resp_deterioration'] == 0) & 
                         (patient_groups['any_cardiac_deterioration'] == 0)).sum()

        # Total number of patients in the DataFrame
        total_patients = len(patient_groups)

        # Calculate percentages for each patient category
        only_resp_percentage = (only_resp_count / total_patients) * 100
        only_cardiac_percentage = (only_cardiac_count / total_patients) * 100
        both_percentage = (both_count / total_patients) * 100
        neither_percentage = (neither_count / total_patients) * 100

        # Calculate total number of windows for each category
        total_windows = len(df)
        windows_only_resp = df[(df['resp_deterioration'] == 1) & (df['cardiac_deterioration'] == 0)].shape[0]
        windows_only_cardiac = df[(df['resp_deterioration'] == 0) & (df['cardiac_deterioration'] == 1)].shape[0]
        windows_both = df[(df['resp_deterioration'] == 1) & (df['cardiac_deterioration'] == 1)].shape[0]
        windows_neither = df[(df['resp_deterioration'] == 0) & (df['cardiac_deterioration'] == 0)].shape[0]

        # Calculate percentages for each window category
        windows_only_resp_percentage = (windows_only_resp / total_windows) * 100
        windows_only_cardiac_percentage = (windows_only_cardiac / total_windows) * 100
        windows_both_percentage = (windows_both / total_windows) * 100
        windows_neither_percentage = (windows_neither / total_windows) * 100

        # Display results for the given DataFrame
        print(f"\n{name} Set - Total Patients: {total_patients}")
        print(f"Patients with Only Respiratory Deterioration: {only_resp_percentage:.2f}%")
        print(f"Patients with Only Cardiac Deterioration: {only_cardiac_percentage:.2f}%")
        print(f"Patients with Both Respiratory and Cardiac Deterioration: {both_percentage:.2f}%")
        print(f"Patients with No Deterioration: {neither_percentage:.2f}%")

        print(f"\n{name} Set - Total Windows: {total_windows}")
        print(f"Windows with Only Respiratory Deterioration: {windows_only_resp_percentage:.2f}%")
        print(f"Windows with Only Cardiac Deterioration: {windows_only_cardiac_percentage:.2f}%")
        print(f"Windows with Both Respiratory and Cardiac Deterioration: {windows_both_percentage:.2f}%")
        print(f"Windows with No Deterioration: {windows_neither_percentage:.2f}%")

    # Check proportions in each set
    calculate_proportion(train_df, "Training")
    calculate_proportion(val_df, "Validation")
    calculate_proportion(test_df, "Test")

    
    ###################################################################################################
    #extract lists of cats_ids
    train_cats_ids = train_df['cats_id'].unique().tolist()
    val_cats_ids = val_df['cats_id'].unique().tolist()
    test_cats_ids = test_df['cats_id'].unique().tolist()

    
    directory = r'/home/workspace/files/'
    csv_files = [f for f in os.listdir(directory) if f.endswith('num.csv')]
    # Create a dictionary with the segment before the first underscore as the key
    patient_dict = {file.split('_')[0]: file for file in csv_files}

    
    # Function to get filenames based on a list of patient IDs
    def get_filenames_from_ids(ids_list, patient_dict):
        return [patient_dict[str(cats_id)] for cats_id in ids_list if str(cats_id) in patient_dict]

    # Generate lists of filenames for each set
    train_filenames = get_filenames_from_ids(train_cats_ids, patient_dict)
    val_filenames = get_filenames_from_ids(val_cats_ids, patient_dict)
    test_filenames = get_filenames_from_ids(test_cats_ids, patient_dict)



    return train_filenames, val_filenames, test_filenames



