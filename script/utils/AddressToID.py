import os
import pandas as pd

# Specify your mapping file name
mapping_file_name = 'Address_to_ID_Mapping.csv'

# Try to load the existing mapping or create a new one if it doesn't exist
if os.path.exists(mapping_file_name):
    address_to_id_df = pd.read_csv(mapping_file_name, index_col='Address')
    address_to_id = address_to_id_df['ID'].to_dict()
else:
    print(f'No such mapping file found -- A new file will be created: {mapping_file_name}')
    address_to_id = {}

next_id = len(address_to_id) + 1  # Next ID to assign


def get_address_id(address):
    global next_id
    if address not in address_to_id:
        address_to_id[address] = next_id
        next_id += 1
    return address_to_id[address]


def replace_addresses_with_ids(df, address_columns):
    for col in address_columns:
        # Vectorize `get_address_id` function call for the entire column
        df[col] = [get_address_id(addr) for addr in df[col]]
    return df


# Insert your folder paths here
folder_path = 'D:\\Token_Data\\SelectedForIDExtraction\\'
folder_output = 'D:\\Token_Data\\pythonProject4\\data\\selected tokens\\1000_wi_tokens'

# List and sort all CSV files by size
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
files_sorted_by_size = sorted(files, key=lambda x: os.path.getsize(os.path.join(folder_path, x)))

processing_indx = 1

for file_name in files_sorted_by_size:
    print(f"Processing {processing_indx} / {len(files_sorted_by_size)}")
    df = pd.read_csv(os.path.join(folder_path, file_name))

    # Assuming the columns are named 'to' and 'from'
    address_columns = ['to', 'from']

    # Replace addresses with IDs
    df = replace_addresses_with_ids(df, address_columns)

    # Save the modified DataFrame
    df.to_csv(os.path.join(folder_output, "wi_" + file_name), index=False)

    print(f"Done with file {processing_indx}")
    processing_indx += 1

# Save the updated address ID mapping after all files are processed
pd.DataFrame(list(address_to_id.items()), columns=['Address', 'ID']).to_csv(mapping_file_name, index=False)

# Save the last processed index
with open('number_file.txt', 'w') as file:
    file.write(str(processing_indx - 1))