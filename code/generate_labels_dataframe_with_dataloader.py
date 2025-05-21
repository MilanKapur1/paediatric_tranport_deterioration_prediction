
from torch.utils.data import DataLoader 
from PatientDataset import PatientDataset
import pandas as pd
from tqdm import tqdm


def generate_labels_dataframe_with_dataloader(patient_list, min_window_min=1, step_min=1,
                                              max_window_min=60, prediction_window_length=15, batch_size=8, num_workers=4, prefetch_factor =2):
    # Initialize the dataset
    dataset = PatientDataset(patient_list=patient_list,
                             min_window_min=min_window_min,
                             step_min=step_min,
                             max_window_min=max_window_min,
                             prediction_window_length=prediction_window_length)
    
    # Create DataLoader with multiple workers
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=True,
                            prefetch_factor=prefetch_factor,
                            pin_memory=False)

    # Initialize an empty DataFrame to store all data
    df = pd.DataFrame(columns=["cats_id", "start_min", "end_min", "resp_deterioration", "cardiac_deterioration"])

    for batch in tqdm(dataloader, desc="Generating labels"):
        # Unpack batch data
        (features,
         missingness_mask,
         padding_mask,
         static_data,
         scalar_tensor,
         list_tensor,
         label_tensor,
         cats_id_tensor,
         start_min_tensor,
         end_min_tensor,
         prediction_window_length) = batch
        
        # Convert tensors to lists or arrays for batch DataFrame
        labels = label_tensor.squeeze().tolist()  # Shape: (batch_size, 2) if each label has two values
        cats_ids = cats_id_tensor.tolist()
        start_mins = start_min_tensor.tolist()
        end_mins = end_min_tensor.tolist()

        # Create DataFrame from batch data
        batch_df = pd.DataFrame({
            "cats_id": cats_ids,
            "start_min": start_mins,
            "end_min": end_mins,
            "resp_deterioration": [label[0] for label in labels],
            "cardiac_deterioration": [label[1] for label in labels]
        })

        # Append to the main DataFrame
        df = pd.concat([df, batch_df], ignore_index=True)

    return df