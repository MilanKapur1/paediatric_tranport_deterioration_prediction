import pandas as pd
import os



def calculate_report(df, dataset_name):
    total_patients = df['cats_id'].nunique()
    total_windows = len(df)

    # Calculate patient-level proportions
    patient_summary = df.groupby('cats_id').agg({
        'resp_deterioration': 'max',
        'cardiac_deterioration': 'max'
    })
    only_resp_patients = ((patient_summary['resp_deterioration'] == 1) & (patient_summary['cardiac_deterioration'] == 0)).mean() * 100
    only_cardiac_patients = ((patient_summary['resp_deterioration'] == 0) & (patient_summary['cardiac_deterioration'] == 1)).mean() * 100
    both_patients = ((patient_summary['resp_deterioration'] == 1) & (patient_summary['cardiac_deterioration'] == 1)).mean() * 100
    no_deterioration_patients = ((patient_summary['resp_deterioration'] == 0) & (patient_summary['cardiac_deterioration'] == 0)).mean() * 100

    # Calculate window-level proportions
    only_resp_windows = ((df['resp_deterioration'] == 1) & (df['cardiac_deterioration'] == 0)).mean() * 100
    only_cardiac_windows = ((df['resp_deterioration'] == 0) & (df['cardiac_deterioration'] == 1)).mean() * 100
    both_windows = ((df['resp_deterioration'] == 1) & (df['cardiac_deterioration'] == 1)).mean() * 100
    no_deterioration_windows = ((df['resp_deterioration'] == 0) & (df['cardiac_deterioration'] == 0)).mean() * 100

    # Create a formatted report string
    report = (
        f"{dataset_name} Set - Total Patients: {total_patients}\n"
        f"Patients with Only Respiratory Deterioration: {only_resp_patients:.2f}%\n"
        f"Patients with Only Cardiac Deterioration: {only_cardiac_patients:.2f}%\n"
        f"Patients with Both Respiratory and Cardiac Deterioration: {both_patients:.2f}%\n"
        f"Patients with No Deterioration: {no_deterioration_patients:.2f}%\n\n"
        f"{dataset_name} Set - Total Windows: {total_windows}\n"
        f"Windows with Only Respiratory Deterioration: {only_resp_windows:.2f}%\n"
        f"Windows with Only Cardiac Deterioration: {only_cardiac_windows:.2f}%\n"
        f"Windows with Both Respiratory and Cardiac Deterioration: {both_windows:.2f}%\n"
        f"Windows with No Deterioration: {no_deterioration_windows:.2f}%\n"
    )
    return report



import os
import pandas as pd

def load_train_test_data(train_filename, val_filename, test_filename, directory='/home/workspace/files/MilanK/Model1/train_test_split'):
    """
    Loads training, validation, and testing patient lists along with labels from specified files.

    Parameters:
    - train_filename (str): Filename for the training patient list.
    - val_filename (str): Filename for the validation patient list.
    - test_filename (str): Filename for the testing patient list.
    - directory (str): Directory where the files are located. Defaults to the original path.

    Returns:
    - tuple: Contains lists for training, validation, and testing patients, and a DataFrame for labels.
    """
    
    # Construct full file paths using os.path.join
    train_path = os.path.join(directory, train_filename)
    val_path = os.path.join(directory, val_filename)
    test_path = os.path.join(directory, test_filename)

    
    # Read the training patient list
    try:
        with open(train_path, 'r') as file:
            train_patient_list = file.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    # Read the validation patient list
    try:
        with open(val_path, 'r') as file:
            val_patient_list = file.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    
    # Read the testing patient list
    try:
        with open(test_path, 'r') as file:
            test_patient_list = file.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Testing file not found: {test_path}")
    
    labels = pd.read_csv('/home/workspace/files/MilanK/Model1/train_test_split/new_labels_15_min_v4.csv')
    
    

    new_train = [int(patient.split('_')[0]) for patient in train_patient_list]
    new_val = [int(patient.split('_')[0]) for patient in val_patient_list]
    new_test = [int(patient.split('_')[0]) for patient in test_patient_list]
    
    train = labels[labels['cats_id'].isin(new_train)]
    val = labels[labels['cats_id'].isin(new_val)]
    test = labels[labels['cats_id'].isin(new_test)]



    train_report = calculate_report(train, "Training")
    val_report = calculate_report(val, "Validation")
    test_report = calculate_report(test, "Testing")

    print(train_report)
    print(val_report)
    print (test_report)
    return train_patient_list, val_patient_list, test_patient_list


