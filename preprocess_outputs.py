import os
import pandas as pd

# Path to the previously created CSV file containing the X data.
# This file should contain the primary dataset to which we want to add the Y data.
x_df_path = './output/prepared_data.csv'  # Replace with the path to your X CSV file

# Load the X CSV file as a DataFrame
# This DataFrame will serve as the base dataset that we will augment with additional Y data
x_df = pd.read_csv(x_df_path)

# Directory containing the Y data files, specifically the "losses_Account_{id}.csv" files
# Only files following this naming convention will be processed, as they contain the relevant data we need.
y_dir = './results'  # Replace with the path to your directory containing Y files

# List to store individual DataFrames for each Y file processed
y_data = []

# Iterate through each file in the specified Y directory
for filename in os.listdir(y_dir):
    # Check if the file follows the 'losses_Account_{id}.csv' naming pattern
    # This pattern ensures we only process files containing account-level loss data
    if filename.startswith('losses_Account_') and filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(y_dir, filename)
        
        # Load the current Y CSV file into a temporary DataFrame
        temp_df = pd.read_csv(file_path)
        
        # Append the loaded DataFrame to the y_data list for later concatenation
        y_data.append(temp_df)

# Concatenate all Y DataFrames into a single DataFrame
# By combining these DataFrames, we obtain a single dataset containing loss data for all accounts
y_df = pd.concat(y_data, ignore_index=True)

# Ensure that the 'accountid' column in y_df matches the naming convention in x_df
# Renaming 'accountid' to 'id' allows us to merge the DataFrames on a common column
y_df = y_df.rename(columns={'accountid': 'id'})

# Merge the x_df and y_df DataFrames on the 'id' column
# We use a left join so that all rows in x_df are preserved, even if there's no matching data in y_df
# This approach ensures that we retain all records from the primary dataset (x_df)
final_df = pd.merge(x_df, y_df, how='left', on='id')

# Display the first few rows of the final merged DataFrame
# This preview helps to verify that the merge was successful and the data was combined correctly
print(final_df.head())

# Optionally, save the final merged DataFrame to a new CSV file
# This file will contain the complete dataset with both X and Y data, ready for analysis or modeling
final_df.to_csv('output/final_data_with_y.csv', index=False)
