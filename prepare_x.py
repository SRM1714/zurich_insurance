import pandas as pd
import json

# Load the original CSV file, which contains the primary dataset with location data stored in JSON format
# The file is loaded into a DataFrame for further processing
df = pd.read_csv('./output/x_data.csv')

# Define a function to expand and extract individual fields from the 'location_info' JSON column
def expand_location_info(row):
    # Convert the JSON string in 'location_info' into a Python dictionary
    # Each 'location_info' field contains a JSON array, so we take the first (and only) element in the list
    location_data = json.loads(row['location_info'])[0]
    
    # Add each field from location_data as a new column in the row
    # This approach flattens the JSON data, making each field accessible as an individual column
    row['location_id'] = location_data.get('location_id', None)
    row['x'] = location_data.get('x', None)  # Longitude coordinate of the location
    row['y'] = location_data.get('y', None)  # Latitude coordinate of the location
    row['construction'] = location_data.get('construction', None)  # Construction type
    row['occupancy'] = location_data.get('occupancy', None)  # Occupancy type
    row['number_floors'] = location_data.get('number_floors', None)  # Number of floors
    row['year_built'] = location_data.get('year_built', None)  # Year the building was constructed
    row['loc_bsum'] = location_data.get('loc_bsum', None)  # Insured amount for this location
    row['loc_bded'] = location_data.get('loc_bded', None)  # Deductible for this location
    row['loc_blim'] = location_data.get('loc_blim', None)  # Limit for this location
    row['country'] = location_data.get('country', None)  # Country where the location is situated
    row['state'] = location_data.get('state', None)  # State or region where the location is situated
    
    # Return the updated row with the new fields added as columns
    return row

# Apply the function to each row in the DataFrame to expand 'location_info'
# The function expands 'location_info' by creating new columns for each key in the JSON data
df = df.apply(expand_location_info, axis=1)

# Remove the original 'location_info' column, as we have extracted all its data into individual columns
df = df.drop(columns=['location_info'])

# Display the processed DataFrame to verify that 'location_info' was expanded correctly
print(df.head())

# Save the processed DataFrame to a new CSV file
# This CSV file now contains separate columns for each field in the original 'location_info' JSON data
df.to_csv('output/prepared_data.csv', index=False)

