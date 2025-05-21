import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import json
import os


def load_static_data (patient_list):    

    # Extract only the patient ID part
    patient_list = [patient.split('_')[0] for patient in patient_list]
    patient_list = [int(patient_id) for patient_id in patient_list]

    # Load the main data file   
    df = pd.read_csv('/home/workspace/files/cats_data_extract_20210920_master_deid_20211019.csv')
    df = df[df['cats_id'].isin(patient_list)]
    df['min_session_time'] = pd.to_datetime(df['min_session_time'], unit='s')

    # Drop unnecessary columns
    df = df.drop(columns=[
        'max_session_time', 'start_datetime', 'end_datetime',
        'length_of_stay', 'length_of_icu_stay', 'deceased_flag',
        'age_at_death', 'referral_outcome', 'referral_outcome_datetime',
        'depart_destunit', 'referral_datetime', 'csv_files',
        'transport_team_in_attendance', 'critical_incident', ' incident_type',
        'ethnicity_name','cats_interv','diagnosis_group','min_session_time'
    ])

    #######################################################################################
    # PIM3 Processing
    # Convert 'pim3' to float and fill NaNs with -1 to represent missing. Pim3 is also logistic score bewteen 0 and 1 so no need to normalise
    #  In some cases pim3 not calculated and Error in PIM3 Score is recorded. This is also set to -1
    
    df['pim3'] = df['pim3'].fillna(-1)
    df['pim3'] = df['pim3'].replace('Error in PIM3 Score', -1)

    df['pim3'] = df['pim3'].astype(float)


    ########################################################################################
    # Encode 'gender' and 'sex' columns consistently

    # Define the mapping from 'gender' to 'sex'
    gender_to_sex_mapping = {'Male': 'M', 'Female': 'F'}

    # Step 1: Map 'gender' to 'sex' format and fill NaNs
    mapped_gender = df['gender'].map(gender_to_sex_mapping)
    df['sex'] = df['sex'].combine_first(mapped_gender)
    df['gender'] = df['gender'].combine_first(df['sex'].map({v: k for k, v in gender_to_sex_mapping.items()}))

    # Step 2: Create unified column based on matching conditions
    df['unified'] = np.where(
        (df['gender'].map(gender_to_sex_mapping) == df['sex']),  # Check if gender and sex match
        df['gender'].map({'Male': 0, 'Female': 1}),              # Encode Male as 0, Female as 1
        -1                                                   # Label mismatches as unknown (-1)
    )
        
    # Step 3: Add a binary indicator for missing or mismatched values in gender
    df['gender_missing'] = np.where(df['unified'] == -1, 1, 0)

    # Step 4: Replace values in 'gender' with values from 'unified'
    df['gender'] = df['unified']

    # Step 5: Drop 'sex' and 'unified' columns
    df.drop(columns=['sex', 'unified'], inplace=True)

    ########################################################################################
    # Age Processing
    #Age at admission: age provided for patients in years but also months, age in year seems to be more
    #precise with decimal points when multiplied up to months. Therefore take age in year, convert to months, use 
    #both values to ensure none missing, and combine to a single age in months column. This will be normalised/scaled later on. 
        
    # Convert 'age_at_admission' to months and combine with 'age_mon'
    df['age_at_admission'] = df['age_at_admission'] * 12  # Convert years to months
    df['age_months'] = df['age_at_admission'].combine_first(df['age_mon'])

    # Step 1: Create a binary indicator for missing values in 'age_months' using np.where
    df['age_months_missing'] = np.where(df['age_months'].isna(), 1, 0)

    # Step 2: Fill NaN values in 'age_months' with -1 for unknowns
    df['age_months'] = df['age_months'].fillna(-1)

    # Drop the original columns now that they are combined
    df.drop(columns=['age_at_admission', 'age_mon'], inplace=True)

    # Step 3: Normalize 'age_months' to a 0-1 range (18 years is 216 months) without affecting the -1 values
    df['age_months'] = np.where(df['age_months'] != -1, df['age_months'] / 216, -1)

    # Reorder columns for clarity
    df = df[[df.columns[0], 'age_months', 'age_months_missing'] + [col for col in df.columns if col not in ['age_months', 'age_months_missing', df.columns[0]]]]

    ########################################################################################
    df['weight_missing'] = df['weight_kg'].isna().astype(int)
    # Set NaNs in 'weight_kg' to -1
    df['weight_kg'] = df['weight_kg'].fillna(-1)


    df['weight_kg'] =  df['weight_kg']/100 # max patient weight say of 100kg so normalise to 0-1 range

    ##########################################################################################
    #Destination Unit: one hot encode destination units. This list of dest units represents 
    #units receiveing more than 50 patients out of the 12000 in full csv list/10 or more patients
    #in the filtered list of 912 patients with at least 15 mins of data

    # Step 1: regex prcoess to handle duplicate labels that are slightly different e.g. extra space/extrahyphen
    df['Destination Unit'] = df['Destination Unit'].dropna().apply(lambda x: x.lower().replace("'", "").replace("-", " ").strip())


    # Step 2: Define unique destinations to retain
    unique_destinations = [
        'london great ormond street hospital   picu_nicu (pic011)',
        'great ormond street hospital',
        'london st marys hospital (pic016)', 
        'cambridge addenbrookes hospital (pic004)',
        'london royal brompton hospital (pic014)',
        'london the royal london hospital (pic032)',
        'london great ormond street hospital   cccu (pic039)',
        'great ormond street hospital central london site (rp401)',
        'great ormond street hospital central london site (rp401) pic011',
        'london evelinas childrens hospital (pic012)',
        'london kings college hospital (pic013)',
        'great ormond street hospital central london site',
        'london st georges hospital (pic015)',
        'other'#this needs to be included to create an other column - even though this value does not exist in the original data, it needs to be generated to facilitate accurate one hot encoding
    ]


    def map_destination(x):
        if isinstance(x, str):
            # Standardize by removing specific suffixes or patterns
            if 'great ormond street hospital' in x:
                return 'great ormond street hospital'
            else:
                return x
        else:
            return x  # Retain NaN or non-string entries as is
    
    df['Destination Unit'] = df['Destination Unit'].apply(map_destination)

    # Step 3: Create 'unknown' column for NaN values
    #df['Destination Unit_missing'] = df['Destination Unit'].isna().astype(int)

    # Step 4: Standardize 'Destination Unit' and set items not in unique_destinations as 'other'
    df['Destination Unit'] = df['Destination Unit'].apply(lambda x: x if x in unique_destinations else 'other' if pd.notna(x) else x)

    # Step 5: One-hot encode 'Destination Unit' with 'other' included
    df = pd.get_dummies(df, columns=['Destination Unit'], prefix='Destination Unit')
    
    
    # Step 6: Ensure all unique destinations have columns, add missing ones if necessary
    onehot_columns = [f'Destination Unit_{dest}' for dest in unique_destinations]
    for col in onehot_columns:
        if col not in df.columns:
            df[col] = 0  # Add the column with 0s if it doesn't exist in the DataFrame


    # Convert the one-hot encoded columns to integers
    onehot_columns = [col for col in df.columns if col.startswith('Destination Unit_')]
    df[onehot_columns] = df[onehot_columns].astype(int)


    ##################################################################################################
    #Referral Unit: one hot encode referral units. This list of dest units represents 
    #units receiveing more than 50 patients out of the 12000 in full csv list



    # Step 1: regex prcoess to handle duplicate labels that are slightly different e.g. extra space/extrahyphen
    df['referring_unit'] = df['referring_unit'].dropna().apply(lambda x: x.lower().replace("'", "").replace("-", " ").strip())


    # Step 2: Define unique destinations to retain
    unique_destinations = ['university college hospital (rrv03)',
                        'northwick park hospital (rv820)',
                        'queens hospital (rf4qh)',
                        'luton and dunstable hospital (rc971)',
                        'basildon university hospital (rddh0)',
                        'chelsea and westminster hospital (rqm01)',
                        'whipps cross university hospital (rgckh)',
                        'north middlesex university hospital (rapnm)',
                        'west middlesex university hospital (rfw01)',
                        'homerton university hospital (rqxm1)',
                        'watford general hospital (rwg02)',
                        'newham general hospital (rnhb1)',
                        'the hillingdon hospital (npv02)',
                        'colchester general hospital (rdee4)',
                        'norfolk and norwich university hospital (rm102)',
                        'barnet hospital (rvl01)', 'royal free hospital (ral01)',
                        'lister hospital (rwh01)', 'the whittington hospital (rkeq4)',
                        'the ipswich hospital (rgq02)', 'southend hospital (raj01)',
                        'bedford hospital south wing (rc110)',
                        'queen charlottes hospital (ryj04)',
                        'peterborough city hospital (rgn80)',
                        'princess alexandra hospital (rqwg0)',
                        'broomfield hospital (rq8l0)',
                        'the royal london hospital (rnj12)',
                        'london the royal london hospital (pic032)',
                        'london great ormond street hospital   picu_nicu (pic011)',
                        'london st marys hospital (pic016)',
                        'james paget university hospital (rgp75)',
                        'hinchingbrooke hospital (rqq31)',
                        'london great ormond street hospital   cccu (pic039)',
                        'west suffolk hospital (rgr50)',               
                        'other'#this needs to be included to create an other column - even though this value does not exist in the original data, it needs to be generated to facilitate accurate one hot encoding
                        ]



    # Step 3: Create 'unknown' column for NaN values
    df['referring_unit_missing'] = df['referring_unit'].isna().astype(int)

    # Step 4: Standardize 'Destination Unit' and set items not in unique_destinations as 'other'
    df['referring_unit'] = df['referring_unit'].apply(lambda x: x if x in unique_destinations else 'other' if pd.notna(x) else x)

    # Step 5: One-hot encode 'Destination Unit' with 'other' included
    df = pd.get_dummies(df, columns=['referring_unit'], prefix='referring_unit')
    
    # Step 6: Ensure all unique destinations have columns, add missing ones if necessary
    onehot_columns = [f'referring_unit_{dest}' for dest in unique_destinations]
    for col in onehot_columns:
        if col not in df.columns:
            df[col] = 0  # Add the column with 0s if it doesn't exist in the DataFrame



    # Convert the one-hot encoded columns to integers
    onehot_columns = [col for col in df.columns if col.startswith('referring_unit_')]
    df[onehot_columns] = df[onehot_columns].astype(int)



    #######################################################################################################################
    # One-hot Encode 'Destination Care Area' with an "Other" category


    # Define categories to retain
    target_categories = ['PICU', 'NICU', 'Ward', 'ICU', 'HDU (step-up / step-down unit)', 'Other']#'other needs to be included to create an other column - even though this value does not exist in the original data, it needs to be generated to facilitate accurate one hot encoding]
    
    #create binary missing column
    df['Destination Care Area_missing'] = df['Destination Care Area'].isna().astype(int)

    # Mark items not in target categories as "Other"
    df['Destination Care Area'] = df['Destination Care Area'].apply(lambda x: x if x in target_categories else 'Other')

    # One-hot encode the 'Destination Care Area' column, including the "Other" category
    df = pd.get_dummies(df, columns=['Destination Care Area'], prefix='Destination Care Area')
    
    
        # Step 6: Ensure all unique destinations have columns, add missing ones if necessary
    onehot_columns = [f'Destination Care Area_{dest}' for dest in target_categories]
    for col in onehot_columns:
        if col not in df.columns:
            df[col] = 0  # Add the column with 0s if it doesn't exist in the DataFrame



    # List of the new one-hot encoded columns
    onehot_columns = [col for col in df.columns if col.startswith('Destination Care Area_')]
    # Convert only the one-hot encoded columns to integers
    df[onehot_columns] = df[onehot_columns].astype(int)



    ###################################################################################################################
    # Encoding 'vasoactive_agent_used' column where only 'Yes' or NaN values are present

    # Convert 'Yes' to 1 and, no to 0 and NaN to -1. Now there aren't any No's recoreded. Only yes /NaN so cant distinguish 
    #between No and NaN but hopefully this handling works
    df['vasoactive_agent_used_misisng'] = df['vasoactive_agent_used'].isna().astype(int)

    df['vasoactive_agent_used'] = df['vasoactive_agent_used'].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else -1))

    #######################################################################################################################
    # One-hot encode specific vasoactive agents listed in 'vasoactive_agent_name'

    # Define the list of unique agents
    unique_agents = ['Adrenaline', 'Dobutamine', 'Dopamine', 'Milrinone', 'Noradrenaline', 'Prostaglandin', 'Vasopressin']

    # Initialize new columns for each unique agent with 0 as the default value
    for agent in unique_agents:
        df[agent] = 0

    # Update the columns based on the presence of each agent in 'vasoactive_agent_name'
    for agent in unique_agents:
        df[agent] = df['vasoactive_agent_name'].apply(lambda x: 1 if pd.notna(x) and agent in x else 0)
        
    df['total_vasoactive_agents'] = df[unique_agents].sum(axis=1)


    # Drop the original 'vasoactive_agent_name' column after encoding
    df.drop(columns=['vasoactive_agent_name'], inplace=True)


    ######################################################################################################################
    # Encoding 'ventilation_status' column to reflect severity of respiratory support with ordered integer values

    #create missing binary column
    df['ventilation_status_missing'] = df['ventilation_status'].isna().astype(int)
    
    # Define the ordered mapping of ventilation statuses
    ventilation_mapping = {
        'Self ventilating (RA)': 1,
        'Self ventilating (supplemental oxygen)': 2,
        'Self ventilating (high flow nasal cannula)': 3,
        'Self ventilating (CPAP)': 4,
        'Self ventilating (BIPAP)': 5,
        'Invasive ventilation (other airway)': 6,
        'Invasive ventilation (ETT)': 7,
        'Invasive ventilation (trachy)': 8
    }

    # Map each status to the integer encoding, with any status not in the mapping or NaN set to -1
    df['ventilation_status_encoded'] = df['ventilation_status'].apply(
        lambda x: ventilation_mapping.get(x, -1)
    )

    # Replace the original 'ventilation_status' column with the encoded values and drop the temporary column
    df['ventilation_status'] = df['ventilation_status_encoded']
    df['ventilation_status'] = df['ventilation_status']/8 #normalise to 0-1 range
    df.drop(columns=['ventilation_status_encoded'], inplace=True)

    ##########################################################################################################
    # Encoding 'inhaled_no' column
    # Binary encoding: 'Yes' to 1, 'No' to 0, and NaN to -1
    df['inhaled_no_missing'] = df['inhaled_no'].isna().astype(int)
    
    df['inhaled_no'] = df['inhaled_no'].map({'Yes': 1, 'No': 0}).fillna(-1)

    ############################################################################################################
    # List of unique interventions from your output
    #I'm guessing all interventions in local intervention column occurred prior to cats team 
    #arriving so fair enough for the model to have knowledge of these, but I'm guessing for cats interventions we can't assume the same? 

    #I.e. We can't assume these were before the data started being collected and hence the model can't know about these as the data may come from sometime during the transport but we can't pinpoint when?

    #again binary missigness column
    df['local_interv_missing'] = df['local_interv'].isna().astype(int)
    unique_interventions = [
        'Peripheral IV access',
        'NGT / OGT',
        'Primary Intubation',
        'Mechanical Ventilation', 
        'Arterial Access',
        'Urinary catheter',
        'Primary Central Venous Access', 
        'Inotrope or Vasopressor Infusion',
        'CT scan',
        'ETT re position', 
        'Prostaglandin Infusion', 
        'Non-Invasive Ventilation',
        'Primary Intraosseous Access',
        'Suction / Physiotherapy', 
        'High Flow Nasal Cannula', 
        'Re-intubation', 'CPR / Defibrillation',
        'Other blood product', 
        'Nitric Oxide',
        'Other Airway', 
        'Chest drain insertion',
        'Packed red blood cells', 
        'Osmotherapy',
        'ETT reposition',
        'Additional Central Venous Access', 
        'Additional Intraosseous Access',
        'Fresh Frozen Plasma',
        'C Spine immobilisation', 
        'Platelet transfusion', 
        'ECMO', 
        'ICP Monitoring',
        'Cryoprecipitate'
    ]

    # Step 1: Initialize new columns for each unique intervention with 0 as the default value
    for intervention in unique_interventions:
        df[intervention] = 0

    # Step 2: Update the columns based on the presence of each intervention in 'local_interv'
    for intervention in unique_interventions:
        df[intervention] = df['local_interv'].apply(lambda x: 1 if pd.notna(x) and intervention in x else 0)
        
    # Step 3: Create 'Other_Intervention' column for interventions not in unique_interventions
    df['Other_Intervention'] = df['local_interv'].apply(
        lambda x: 1 if pd.notna(x) and any(interv not in unique_interventions for interv in x.split(',')) else 0
    )

    # Optional: Add a column to count total interventions for each row (patient)
    df['total_interventions'] = df[unique_interventions].sum(axis=1)

    # Drop the original 'local_interv' column after encoding, if no longer needed
    df.drop(columns=['local_interv'], inplace=True)


    ###########################################################################################################################
    ## One hot code ethnicity. Have dropped ethnicity name as letter code is mapped using:
    #https://digital.nhs.uk/data-and-information/data-collections-and-data-sets/data-sets/mental-health-services-data-set/submit-data/data-quality-of-protected-characteristics-and-other-vulnerable-groups/ethnicity
    # and easier to use letters due to regex

    df['ethnicity_category_missing'] = df['ethnicity_nat_code'].isna().astype(int)
    # Define the mapping for ethnicity codes
    ethnicity_dict = {
        'A': 'White - British',
        'B': 'White - Irish',
        'C': 'White - Any other White background',
        'D': 'Mixed - White and Black Caribbean',
        'E': 'Mixed - White and Black African',
        'F': 'Mixed - White and Asian',
        'G': 'Mixed - Any other mixed background',
        'H': 'Asian or Asian British - Indian',
        'J': 'Asian or Asian British - Pakistani',
        'K': 'Asian or Asian British - Bangladeshi',
        'L': 'Asian or Asian British - Any other Asian background',
        'M': 'Black or Black British - Caribbean',
        'N': 'Black or Black British - African',
        'P': 'Black or Black British - Any other Black background',
        'R': 'Other Ethnic Groups - Chinese',
        'S': 'Other Ethnic Groups - Any other ethnic group',
        'Z': 'Not stated',
        '99': 'Unknown'  # Assign 'Unknown' for code '99'
    }

    # Replace NaN and '99' with 'Unknown'
    df['ethnicity_category'] = df['ethnicity_nat_code'].map(ethnicity_dict)

    # One-hot encode the mapped ethnicity categories
    df = pd.concat([df, pd.get_dummies(df['ethnicity_category'], prefix='ethnicity')], axis=1)

    df.drop(columns=['ethnicity_nat_code', 'ethnicity_category'], inplace=True)
    
    
    
    # Step 6: Ensure all unique destinations have columns, add missing ones if necessary
    onehot_columns = [f'ethnicity_{ethnicity_dict[ethnicity]}' for ethnicity in ethnicity_dict]
    for col in onehot_columns:
        if col not in df.columns:
            df[col] = 0  # Add the column with 0s if it doesn't exist in the DataFrame



    
    

    # List of the new one-hot encoded columns
    ethnicity_columns = [col for col in df.columns if col.startswith('ethnicity_')]

    # Convert only the one-hot encoded columns to integers
    df[ethnicity_columns] = df[ethnicity_columns].astype(int)



    ###########################################################################################################################
    ## one hot encode the pre-existing diagnosis into the avalable categories.
    #NaN are set to unknown as not None or other which are options in the list

    # create binary missing column    
    df['preexisting_conditions_missing'] = df['preexisting_conditions'].isna().astype(int)
    
    
    # Define the list of unique agents, including 'Unknown'
    pmh = ['Neurological', 'Cardiac', 'Respiratory', 'Multi-system', 'Genetic / Syndrome',
        'Metabolic / Endocrine', 'Haem / Onc', 'Other', 'Renal', 'None']

    # Initialize new columns for each unique condition with 0 as the default value
    for existing_condition in pmh:
        df[existing_condition] = 0

    # Update the columns based on the presence of each condition in 'preexisting_conditions'
    for existing_condition in pmh:
        df[existing_condition] = df['preexisting_conditions'].apply(
            lambda x: 1 if pd.notna(x) and existing_condition in x else 0
        )

    # Create a column to count total pre-existing conditions per row
    columns_to_sum = [col for col in pmh if col not in ['None', 'Unknown']]
    # Calculate the sum, excluding 'None' and 'Unknown'
    df['total_pre_existing_condition'] = df[columns_to_sum].sum(axis=1)



    # Drop the original 'preexisting_conditions' column after encoding
    df.drop(columns=['preexisting_conditions'], inplace=True)



    ####################################################################################################################

    ## Day Night divide: This section takes the time the team arrived at the collection unit and if between 9am and 9pm labels as day shift
    # if 9pm to 9am night shift. then drops original time column. THe idea here is that patients overnight perhaps might be more unstable??

    #day set to 0, night to 1 and unknown to -1

    df['arrive_collunit'] = pd.to_datetime(df['arrive_collunit'], format='%d/%m/%Y %H:%M', errors='coerce')


    # Create 'day_night' column based on the time in 'arrive_collunit'
    # Define the time boundaries
    day_start = pd.to_datetime("09:00:00").time()
    day_end = pd.to_datetime("21:00:00").time()

    # Create 'day_night' column based on the time in 'arrive_collunit', setting NaT values to -1
    df['day_night'] = df['arrive_collunit'].apply(lambda x: 0 if pd.notna(x) and day_start <= x.time() < day_end else (1 if pd.notna(x) else -1))
    df.drop(columns=['arrive_collunit'], inplace=True)

    #######################################################################################################################

    #embed primary diagnosis to vector embedding using bioclinical bert. 
    #THis is necessary as the diagnosis group column is sparse and not accurate. 
    # and the Primary_diagnosis column has like many hundred unique entries all similar 
    # some are categorised well e.g. 600 acute bronchiolitis out of the 12000 in full csv
    # however many are rarer and need a way to handle new unique values in future - hence vecotr embed.

    #this script does this on demand as needed if not in cache of known embeddings.
    # uses biolclincal bert downloaded form huggungface: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
    # as this embedder has been finetuned on ICU data from MIMIC III (a database containing electronic 
    #health records from ICU patients at the Beth Israel Hospital in Boston, MA) so will hopefully handle 
    #this patient set better


    #binary missing column
    df['primary_diagnosis_missing'] = df['primary_diagnosis'].isna().astype(int)

        # Step 1: Replace NaNs with 'Unknown'    
    df['primary_diagnosis'] = df['primary_diagnosis'].fillna('Unknown')


    # Step 2: Get unique diagnoses
    diagnoses = df['primary_diagnosis'].unique()

    # Step 3: Load or initialize the cache
    cache_file = '/home/workspace/files/MilanK/code/embedding_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    # Step 4: Identify missing diagnoses that need embeddings
    diagnoses = df['primary_diagnosis'].unique()
    missing_diagnoses = [diag for diag in diagnoses if diag not in cache]




    # Only load the tokenizer and model if there are missing diagnoses
    if missing_diagnoses:

        # Specify the path to your cloned repository
        model_path = './code/Bio_ClinicalBERT'
        # Load the tokenizer and model from the local directory
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()  # Set the model to evaluation mode
        
        
        # Step 5: Function to compute embeddings
        def get_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.pooler_output.squeeze().numpy()
            return embedding

        # Step 6: Compute embeddings and update the cache
        for diagnosis in missing_diagnoses:
            embedding = get_embedding(diagnosis)
            cache[diagnosis] = embedding.tolist()  # Convert to list for JSON serialization

            
        # Step 7: Save the updated cache
        with open(cache_file, 'w') as f:
            json.dump(cache, f)

    # Step 8: Map embeddings back to the DataFrame
    def get_cached_embedding(diagnosis):
        return np.array(cache[diagnosis])

    df['primary_diagnosis_embedding'] = df['primary_diagnosis'].apply(get_cached_embedding)

    df.drop(columns=['primary_diagnosis'], inplace=True)
    
    
    
    #######################################################################################################################
    final_column_order =   ['cats_id',
                            'age_months',
                            'age_months_missing',
                            'gender',
                            'gender_missing',
                            'weight_kg',
                            'weight_missing',
                            'ventilation_status',
                            'ventilation_status_missing',
                            'pim3',
                            # 'Destination Unit_cambridge addenbrookes hospital (pic004)',
                            # 'Destination Unit_great ormond street hospital',
                            # 'Destination Unit_london evelinas childrens hospital (pic012)',
                            # 'Destination Unit_london kings college hospital (pic013)',
                            # 'Destination Unit_london royal brompton hospital (pic014)',
                            # 'Destination Unit_london st georges hospital (pic015)',
                            # 'Destination Unit_london st marys hospital (pic016)',
                            # 'Destination Unit_london the royal london hospital (pic032)',
                            # 'Destination Unit_other',
                            # 'referring_unit_missing',
                            # 'referring_unit_barnet hospital (rvl01)',
                            # 'referring_unit_basildon university hospital (rddh0)',
                            # 'referring_unit_bedford hospital south wing (rc110)',
                            # 'referring_unit_broomfield hospital (rq8l0)',
                            # 'referring_unit_chelsea and westminster hospital (rqm01)',
                            # 'referring_unit_colchester general hospital (rdee4)',
                            # 'referring_unit_hinchingbrooke hospital (rqq31)',
                            # 'referring_unit_homerton university hospital (rqxm1)',
                            # 'referring_unit_james paget university hospital (rgp75)',
                            # 'referring_unit_lister hospital (rwh01)',
                            # 'referring_unit_london great ormond street hospital   cccu (pic039)',
                            # 'referring_unit_london great ormond street hospital   picu_nicu (pic011)',
                            # 'referring_unit_london st marys hospital (pic016)',
                            # 'referring_unit_london the royal london hospital (pic032)',
                            # 'referring_unit_luton and dunstable hospital (rc971)',
                            # 'referring_unit_newham general hospital (rnhb1)',
                            # 'referring_unit_norfolk and norwich university hospital (rm102)',
                            # 'referring_unit_north middlesex university hospital (rapnm)',
                            # 'referring_unit_northwick park hospital (rv820)',
                            # 'referring_unit_peterborough city hospital (rgn80)',
                            # 'referring_unit_princess alexandra hospital (rqwg0)',
                            # 'referring_unit_queen charlottes hospital (ryj04)',
                            # 'referring_unit_queens hospital (rf4qh)',
                            # 'referring_unit_royal free hospital (ral01)',
                            # 'referring_unit_southend hospital (raj01)',
                            # 'referring_unit_the hillingdon hospital (npv02)',
                            # 'referring_unit_the ipswich hospital (rgq02)',
                            # 'referring_unit_the royal london hospital (rnj12)',
                            # 'referring_unit_the whittington hospital (rkeq4)',
                            # 'referring_unit_university college hospital (rrv03)',
                            # 'referring_unit_watford general hospital (rwg02)',
                            # 'referring_unit_west middlesex university hospital (rfw01)',
                            # 'referring_unit_west suffolk hospital (rgr50)',
                            # 'referring_unit_whipps cross university hospital (rgckh)',
                            # 'referring_unit_other',
                            'Destination Care Area_missing',
                            'Destination Care Area_HDU (step-up / step-down unit)',
                            'Destination Care Area_ICU',
                            'Destination Care Area_NICU',                            
                            'Destination Care Area_PICU',
                            'Destination Care Area_Ward',
                            'Destination Care Area_Other',
                            'vasoactive_agent_used',
                            'vasoactive_agent_used_misisng',                           
                            'Adrenaline',
                            'Dobutamine',
                            'Dopamine',
                            'Milrinone',
                            'Noradrenaline',
                            'Prostaglandin',
                            'Vasopressin',
                            'total_vasoactive_agents',
                            # 'inhaled_no',
                            # 'inhaled_no_missing',
                            # 'Peripheral IV access',
                            # 'NGT / OGT',
                            # 'Primary Intubation',
                            # 'Mechanical Ventilation', 
                            # 'Arterial Access',
                            # 'Urinary catheter',
                            # 'Primary Central Venous Access', 
                            # 'Inotrope or Vasopressor Infusion',
                            # 'CT scan',
                            # 'ETT re position', 
                            # 'Prostaglandin Infusion', 
                            # 'Non-Invasive Ventilation',
                            # 'Primary Intraosseous Access',
                            # 'Suction / Physiotherapy', 
                            # 'High Flow Nasal Cannula', 
                            # 'Re-intubation', 'CPR / Defibrillation',
                            # 'Other blood product', 
                            # 'Nitric Oxide',
                            # 'Other Airway', 
                            # 'Chest drain insertion',
                            # 'Packed red blood cells', 
                            # 'Osmotherapy',
                            # 'ETT reposition',
                            # 'Additional Central Venous Access', 
                            # 'Additional Intraosseous Access',
                            # 'Fresh Frozen Plasma',
                            # 'C Spine immobilisation', 
                            # 'Platelet transfusion', 
                            # 'ECMO', 
                            # 'ICP Monitoring',
                            # 'Cryoprecipitate',
                            # 'total_interventions',
                            # 'local_interv_missing',
                            # 'ethnicity_Asian or Asian British - Any other Asian background',
                            # 'ethnicity_Asian or Asian British - Bangladeshi',
                            # 'ethnicity_Asian or Asian British - Indian',
                            # 'ethnicity_Asian or Asian British - Pakistani',
                            # 'ethnicity_Black or Black British - African',
                            # 'ethnicity_Black or Black British - Any other Black background',
                            # 'ethnicity_Black or Black British - Caribbean',
                            # 'ethnicity_Mixed - Any other mixed background',
                            # 'ethnicity_Mixed - White and Asian',
                            # 'ethnicity_Mixed - White and Black African',
                            # 'ethnicity_Mixed - White and Black Caribbean',
                            # 'ethnicity_Not stated',
                            # 'ethnicity_Other Ethnic Groups - Any other ethnic group',
                            # 'ethnicity_Other Ethnic Groups - Chinese',
                            # 'ethnicity_Unknown',
                            # 'ethnicity_White - Any other White background',
                            # 'ethnicity_White - British',
                            # 'ethnicity_White - Irish',
                            # 'ethnicity_category_missing',
                            'Neurological',
                            'Cardiac',
                            'Respiratory',
                            'Multi-system',
                            'Genetic / Syndrome',
                            'Metabolic / Endocrine',
                            'Haem / Onc',
                            'Other',
                            'Renal',
                            'None',
                            'preexisting_conditions_missing',
                            'total_pre_existing_condition',
                            'day_night',
                            'primary_diagnosis_embedding',
                            'primary_diagnosis_missing']
    
    df = df [final_column_order]

    return df

    

