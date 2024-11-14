
import numpy as np
import csv
import os
import json
import matplotlib.pyplot as plt
import time
import pandas as pd

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from sklearn.model_selection import train_test_split

import GPy

import multiprocessing as mp
from tqdm import tqdm


def plot(xs, ys, values, vmin,vmax,figname):

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(xs, ys, c=values, cmap='viridis', vmin=vmin, vmax=vmax, marker='o', s=5)  # Change marker size with 's' parameter
    plt.colorbar(sc, label='Targets')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('2D Color Plot of Property Value')
    plt.savefig(figname)

    return 0

def read_csv_losses(account_id):
    filename = os.path.join('results', f'losses_Locations_{account_id}.csv')
    data = []

    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    
    return data

def read_json_inputs(account_id):
    filename = os.path.join('smallAccount', f'account_{account_id.astype(int)}.json')
    data = []

    with open(filename, 'r') as file:
        data = json.load(file)
    
    return data

def get_locations(data):
    if 'exposure' in data and 'locations' in data['exposure']:
        return data['exposure']['locations']
    return []

def get_periods(data, location_index):
    periods_data = []
    if 'hazard' in data and 'perils' in data['hazard']:
        for peril in data['hazard']['perils']:
            if 'locations' in peril:
                location = peril['locations'][location_index]
                if 'attributes_rp' in location and isinstance(location['attributes_rp'], list):
                    for attribute in location['attributes_rp']:
                        if 'return_periods' in attribute and isinstance(attribute['return_periods'], list):
                            for period in attribute['return_periods']:
                                if 'value' in period:
                                    periods_data.append(period['value'])
                                else:
                                    print(f"Warning: 'value' not found in period {period}")
                        else:
                            print(f"Warning: 'return_periods' not found or not a list in attribute {attribute}")
                else:
                    print(f"Warning: 'attributes_rp' not found or not a list in location {location}")
    return periods_data

# def process_result(result):
#     target = result[2]  # Ground up loss
#     location_idx = result[1].astype(int)  # Location ID
#     accountid = result[0]  # Account ID
#     data = read_json_inputs(accountid)
#     location_dict = get_locations(data)[location_idx]
#     hazards_dict = get_periods(data, location_idx)
#     x = location_dict["x"]
#     y = location_dict["y"]
#     value = location_dict["bsum"]
#     hazard_values = [float(hazards_dict[n]) for n in range(len(hazards_dict))]
#     input = [x, y, value]
#     return input, target, hazard_values

def process_result(result):
    target = result[2]  # Ground up loss
    location_idx = result[1].astype(int)  # Location ID
    accountid = result[0]  # Account ID

    # Read JSON data for the specified account
    data = read_json_inputs(accountid)
    
    # Retrieve location information and hazard data for the specified location index
    locations = get_locations(data)
    location_dict = locations[location_idx]
    hazards_dict = get_periods(data, location_idx)

    # Extract coordinates, value, and other location information
    x = location_dict["x"]
    y = location_dict["y"]
    value = location_dict["bsum"]

    # Calculate hazard values and count of locations for the account
    hazard_values = [float(hazards_dict[n]) for n in range(len(hazards_dict))]
    location_count = len(locations)  # Total number of locations for this account

    # Add location count to the input
    input_data = [x, y, value, location_count]

    # Return input data, target, and hazard values
    return input_data, target, hazard_values


def process_result2(result):
    target = result[2]  # Ground up loss
    location_idx = result[1].astype(int)  # Location ID
    accountid = result[0]  # Account ID
    data = read_json_inputs(accountid)
    location_dict = get_locations(data)[location_idx]
    hazards_dict = get_periods(data, location_idx)
    x = location_dict["x"]
    y = location_dict["y"]
    value = location_dict["bsum"]
    hazard_values = [float(hazards_dict[n]) for n in range(len(hazards_dict))]
    input_data = [accountid, location_idx, x, y, value, target]
    return input_data, target, hazard_values


def main():

    results_file = 'results.npy'

    print("Loading results...", flush=True)

    t0 = time.time()

    if os.path.exists(results_file):
        # Load the results from the file
        results = np.load(results_file, allow_pickle=True)
        account_indexes = np.load('account_indexes.npy', allow_pickle=True)
    else:
        results = []
        account_indexes = []

        for i in range(50000):
            data = read_csv_losses(i)
            num_of_locations = len(data)  # Number of locations

            num_of_locations = len(data) # Number of locations
            account_location_index = len(results)

            account_indexes.append([account_location_index, account_location_index + num_of_locations]) #saves start and end index of locations for each account
                

            for k, row in enumerate(data):
                accountid = row['accountid']  # Account ID
                locationidx = k  # Location index
                aal_gu = row['AAL_GU']  # AAL_GU is the ground-up loss
                aal_gr = row['AAL_GR']  # AAL_GR is the gross loss

                results.append([accountid, locationidx, aal_gu, aal_gr])

        results = np.array(results, dtype=float)  # Convert results to a numpy array
        np.save(results_file, results)  # Save results to a file
        np.save('account_indexes.npy', account_indexes)  # Save account indexes to a file

    print(results, flush=True)
    t1 = time.time()
    print("Time to load results:", t1-t0, flush=True)

    accounts = 1000

    stop = account_indexes[accounts-1][1]

    results = results[:stop]

    print(f"Number of buildings: {len(results)}")

    inputs = []
    targets = []
    hazards = []

    pool = mp.Pool(mp.cpu_count())
    processed_results = list(tqdm(pool.imap(process_result, results), total=len(results)))

    inputs, targets, hazards = zip(*processed_results)

    processed_results2 = list(tqdm(pool.imap(process_result2, results), total=len(results)))
    initial_info, a, b = zip(*processed_results2)
    initial_info = np.array(initial_info, dtype=object)  # 2D array for initial info
        
    inputs = np.array(inputs, dtype=float)
    targets = np.array(targets,dtype=float)
    hazards = np.array(hazards,dtype=float)
    print("Done processing results.")

    x0, x1 = -80.7, -80
    y0, y1 = 25.5, 27

    xs = inputs[:,0]
    ys = inputs[:,1]
    values = inputs[:,2]
    num_locations = inputs[:,3]
    targets = targets[:]
    rel_targets = targets / values

    mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

    xs = xs[mask]
    ys = ys[mask]
    values = values[mask]
    num_locations = num_locations[mask]
    rel_targets = rel_targets[mask]
    hazards = hazards[mask]

    plot(xs, ys, rel_targets, np.min(rel_targets), np.max(rel_targets), "Targets")

    # Prepare the data for training
    X = np.column_stack((xs, ys, num_locations, hazards))

    print("X shape", X.shape)

    y = rel_targets

    print("y shape", y.shape)

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_training_accounts = 400
    num_testing_accounts = 100

    #training_idx = account_indexes[num_training_accounts-1][1]
    #testing_idx = account_indexes[num_training_accounts+num_testing_accounts-1][1]

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    training_idx = 1000

    X_train = X_shuffled[:training_idx]
    y_train = y_shuffled[:training_idx]
    X_test = X_shuffled[training_idx:]
    y_test = y_shuffled[training_idx:]

    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("y_train shape", y_train.shape)

    # Convert data to PyTorch tensors
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64).reshape(-1,1)
    y_test = np.array(y_test, dtype=np.float64).reshape(-1, 1)


    ######################################## model to predict loss values from hazard values ########################################

    # Define the Gaussian Process model
    print('aaaaaaa')
    print(X_train[1])
    kernel = GPy.kern.RBF(input_dim=10, variance=1, lengthscale=1)
    model = GPy.models.GPRegression(X_train[:,2:], np.log(y_train), kernel)

    print("Optimizing the model...")
    # Optimize the model
    model.optimize(messages=True)

    y_pred, y_pred_var = model.predict(X_test[:,2:])

    Error = np.linalg.norm(np.exp(y_pred) - y_test)/np.linalg.norm(y_test) *100

    print("Error from periods prediction:", Error)

    y_test = np.array(y_test).flatten()
    y_pred = np.exp(np.array(y_pred).flatten())

    individual_errors = np.abs(y_pred - y_test) / np.abs(y_test) * 100
    threshold = 15
    exceeding_indices = np.where(individual_errors > threshold)[0]

    # Retrieve initial information for errors exceeding threshold
    exceeding_initial_info = initial_info[training_idx:][exceeding_indices]
    exceeding_data = pd.DataFrame(exceeding_initial_info, columns=['AccountID', 'LocationIdx', 'X', 'Y', 'PropertyValue', 'Target'])
    exceeding_data['Predicted'] = y_pred[exceeding_indices]
    exceeding_data['Actual'] = y_test[exceeding_indices]
    exceeding_data['Percentage_Error'] = individual_errors[exceeding_indices]

    # Display the filtered data
    print(exceeding_data)
    num_unique_account_ids = exceeding_data['AccountID'].nunique()

    # Display the result
    print(f"Number of unique AccountID values: {num_unique_account_ids}")
    top_3_account_ids = exceeding_data['AccountID'].value_counts().head(3)

    # Display the result
    print("The 3 most repeated AccountID values and their counts:")
    print(top_3_account_ids)

    top_3_account_ids = exceeding_data['AccountID'].value_counts().head(3).index

    # Filter the DataFrame to include only rows where AccountID is in the top 3
    top_accounts_df = exceeding_data[exceeding_data['AccountID'].isin(top_3_account_ids)]

    # Print the resulting DataFrame
    print(top_accounts_df)
    #exceeding_data.to_csv('exceeding_15_percent_error_detailed.csv', index=False)

    # vmin = y_test.min()
    # vmax = y_test.max()
    

    # plot(X_test[:,0], X_test[:,1], np.exp(y_pred),vmin,vmax, "Predictions Relative Loss")
    # plot(X_test[:,0], X_test[:,1], y_test,vmin,vmax, "Ground Truth Relative Loss")

    """

    y_abs_pred = y_pred*value_test.reshape(-1,1)
    y_abs = y_test*value_test.reshape(-1,1)

    vmin = y_abs.min()
    vmax = y_abs.max()

    plot(X_test[:,0], X_test[:,1], y_abs_pred,vmin,vmax, "Predictions Absolute Loss")
    plot(X_test[:,0], X_test[:,1], y_abs,vmin,vmax, "Ground Truth Absolute Loss")
    """

if __name__ == '__main__':
    main()