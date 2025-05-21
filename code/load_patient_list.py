
import os
def load_patient_list(file_name):
    directory = r'/home/workspace/files/MilanK/Model1'
    file_path = os.path.join(directory, file_name)

    # Load the file contents into a list
    with open(file_path, 'r') as file:
        patient_list = [line.strip() for line in file]  # Remove newline characters from each line
        
    id_list = []
    for filename in patient_list:
        cats_id = int(filename.split('_')[0]) 
        id_list.append(cats_id)
    print(len(id_list))


    unique_ids = set(id_list)
    # Get the number of unique values
    unique_count = len(unique_ids)
    print(f"Number of unique values: {unique_count}")
    
    return patient_list
