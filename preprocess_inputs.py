import os
import json
import pandas as pd

# Directory containing the JSON files for X data
# Each file in this directory follows the naming pattern 'account_{id}.json'
x_dir = './smallAccount'  # Replace with the path to your directory

# List to store processed data for each JSON file
# Each JSON file will be read, and its data will be appended to this list as a dictionary
x_data = []

# Process each JSON file in the X directory
for filename in os.listdir(x_dir):
    # Check if the file follows the naming convention 'account_{id}.json'
    # Only files with this pattern will be processed
    if filename.endswith('.json') and filename.startswith('account_'):
        # Extract the account ID from the filename (e.g., 'account_0.json' -> 0)
        # This ID will serve as a unique identifier for each record in the final DataFrame
        account_id = int(filename.split('_')[1].split('.')[0])
        
        # Load the JSON file data
        with open(os.path.join(x_dir, filename), 'r') as f:
            data = json.load(f)
        
        # Extract the relevant information from the JSON data
        # 'exposure' contains basic account and location-related information
        exposure = data.get('exposure', {})
        
        # 'hazard' contains information on perils (e.g., windstorm) and return period data
        # We only take the first item in the 'perils' list
        hazard = data.get('hazard', {}).get('perils', [])[0]
        
        # Extract account-level exposure data (general information about the insured account)
        peril = exposure.get('peril', '')
        peril_region = exposure.get('peril_region', '')
        bsum = exposure.get('bsum', 0)  # Total insured amount for the account
        bded = exposure.get('bded', 0)  # Deductible amount for the account
        blim = exposure.get('blim', 0)  # Maximum insured limit for the account
        currency = exposure.get('account_currency', 'USD')  # Currency used in the account
        line_of_business = exposure.get('line_of_business', '')  # Business line identifier

        # Process location-specific information and store it as a list of dictionaries
        location_info = []
        for loc in exposure.get('locations', []):
            # Append each location's details as a dictionary to the location_info list
            location_info.append({
                'location_id': loc.get('locationId', ''),
                'x': loc.get('x', None),  # Longitude coordinate
                'y': loc.get('y', None),  # Latitude coordinate
                'construction': loc.get('construction', ''),  # Construction type
                'occupancy': loc.get('occupancy', ''),  # Occupancy type
                'number_floors': loc.get('number_floors', 0),  # Number of floors in the building
                'year_built': loc.get('year_built', None),  # Year the building was constructed
                'loc_bsum': loc.get('bsum', 0),  # Insured amount for this location
                'loc_bded': loc.get('bded', 0),  # Deductible for this location
                'loc_blim': loc.get('blim', 0),  # Insured limit for this location
                'country': loc.get('country', ''),  # Country of the location
                'state': loc.get('state', '')  # State or region of the location
            })
        
        # Convert location_info (a list of dictionaries) to a JSON string
        # This allows us to store all location data in a single column in the DataFrame
        location_info_str = json.dumps(location_info)

        # Extract return period data (risk values associated with different return periods)
        # The return periods are stored in 'attributes_rp' within each hazard location
        return_period_data = {}
        for hazard_attr in hazard.get('locations', []):
            for rp in hazard_attr['attributes_rp'][0].get('return_periods', []):
                # Create a column for each return period with the format "RP_{period}"
                # The period value serves as a unique identifier for each risk value
                return_period_data[f"RP_{rp['period']}"] = float(rp['value'])

        # Append a dictionary representing the processed data for this file to x_data
        # Each dictionary represents a row in the final DataFrame, with all relevant fields
        x_data.append({
            'id': account_id,  # Unique identifier for each account based on filename
            'peril': peril,
            'peril_region': peril_region,
            'bsum': bsum,
            'bded': bded,
            'blim': blim,
            'currency': currency,
            'line_of_business': line_of_business,
            'location_info': location_info_str,  # Store location data as a JSON string
            **return_period_data  # Unpack return period columns for this account
        })

# Convert the list of dictionaries (x_data) to a DataFrame
# This DataFrame contains one row per JSON file, with columns for each extracted field
x_df = pd.DataFrame(x_data)

# Display the first few rows of the resulting DataFrame for verification
print(x_df.head())

# Save the final DataFrame to a CSV file (optional)
# This CSV file can be used in further data processing or analysis
x_df.to_csv('output/x_data.csv', index=False)
