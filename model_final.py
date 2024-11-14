
import numpy as np
import csv
import os
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

import GPy

import multiprocessing as mp
from tqdm import tqdm

def plot(xs, ys, values, vmin,vmax,figname, cmap='viridis'):

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(xs, ys, c=values, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=5)  # Change marker size with 's' parameter
    plt.colorbar(sc, label='Targets')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title(figname)
    # Ensure the directory exists
    os.makedirs('Figures', exist_ok=True)

    # Save the figure in the specified folder
    plt.savefig(os.path.join('Figures', figname), dpi=100)
    plt.close()

    return 0

def plot_v2(xs, ys, values, vmin,vmax,figname, xlim, ylim, cmap='viridis'):

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(xs, ys, c=values, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=5)  # Change marker size with 's' parameter
    plt.colorbar(sc, label='Targets')
    plt.xlabel('Ground Up Loss')
    plt.ylabel('Gross Loss')
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Ensure the directory exists
    os.makedirs('Figures', exist_ok=True)

    # Save the figure in the specified folder
    plt.savefig(os.path.join('Figures', figname), dpi=100)
    plt.close()

    return 0

def plot_2d(AAL_GU, AAL_GR, figname):

    plt.figure(figsize=(10, 8))
    plt.scatter(AAL_GU, AAL_GR, marker='o', s=5)  # Change marker size with 's' parameter
    plt.xlabel('Ground Up Loss')
    plt.ylabel('Gross Loss')
    # Ensure the directory exists
    os.makedirs('Figures', exist_ok=True)

    # Save the figure in the specified folder
    plt.savefig(os.path.join('Figures', figname))
    plt.close()

    return 0

def read_csv_losses(account_id):
    filename = os.path.join('results', f'losses_Locations_{account_id}.csv')
    data = []

    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    
    return data

def get_account_losses(account_id):
    filename = os.path.join('results', f'losses_Account_{account_id}.csv')
    data = []

    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    
    for k, row in enumerate(data):
                accountid = row['accountid']
                aal_gr = row['AAL_GR']  # AAL_GR is the gross loss

    return [accountid, aal_gr]  

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

def process_result(result):
    AAL_GR = result[3]  # Gross loss
    AAL_GU = result[2]  # Ground up loss
    num_of_locations = result[4]
    location_idx = result[1].astype(int)  # Location ID
    accountid = result[0]  # Account ID
    data = read_json_inputs(accountid)
    location_dict = get_locations(data)[location_idx]
    hazards_dict = get_periods(data, location_idx)
    x = location_dict["x"]
    y = location_dict["y"]
    value = location_dict["bsum"]
    deductible = location_dict["bded"]
    limit = location_dict["blim"]
    hazard_values = [float(hazards_dict[n]) for n in range(len(hazards_dict))]
    params = [x, y, value, deductible, limit]
    return params, AAL_GU, AAL_GR, hazard_values, num_of_locations


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
            account_location_index = len(results)
            account_indexes.append([account_location_index, account_location_index + num_of_locations]) #saves start and end index of locations for each account
            
            for k, row in enumerate(data):
                accountid = row['accountid']  # Account ID
                locationidx = k  # Location index
                aal_gu = row['AAL_GU']  # AAL_GU is the ground-up loss
                aal_gr = row['AAL_GR']  # AAL_GR is the gross loss

                results.append([accountid, locationidx, aal_gu, aal_gr, num_of_locations])

        results = np.array(results, dtype=float)  # Convert results to a numpy array
        np.save(results_file, results)  # Save results to a file
        np.save('account_indexes.npy', account_indexes)  # Save account indexes to a file

    t1 = time.time()
    print("Time to load results:", t1-t0, flush=True)

    accounts = 10000

    stop = account_indexes[accounts-1][1]

    results_testing = results.copy()

    results = results[:stop]


    print(f"Number of buildings: {len(results)}")

    params = []
    inputs = []
    targets = []
    hazards = []
    num_of_locations = []

    pool = mp.Pool(mp.cpu_count())
    processed_results = list(tqdm(pool.imap(process_result, results), total=len(results)))

    params, inputs, targets, hazards, num_of_locations = zip(*processed_results)
        
    params = np.array(params, dtype=float)
    inputs = np.array(inputs, dtype=float)
    targets = np.array(targets,dtype=float)
    hazards = np.array(hazards,dtype=float)
    num_of_locations = np.array(num_of_locations, dtype=float)

    print("Done processing results.")

    x0, x1 = -80.7, -80
    y0, y1 = 25.5, 27

    xs = params[:,0]
    ys = params[:,1]
    values = params[:,2]
    bdeds = params[:,3]
    blims = params[:,4]
    targets = targets[:]
    inputs = inputs[:]

    mask_data = False

    if mask_data:

        mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

        xs = xs[mask]
        ys = ys[mask]
        values = values[mask]
        targets = targets[mask]
        inputs = inputs[mask]
        bdeds = bdeds[mask]
        blims = blims[mask]
        hazards = hazards[mask]
        num_of_locations = num_of_locations[mask]


    # Prepare the data for training
    X = np.column_stack((xs, ys, hazards, bdeds, blims, values, num_of_locations))
    y = targets

    indices = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]
    values_shuffled = values[indices]

    training_idx = len(X_shuffled) - 5000

    X_train = X_shuffled[:training_idx]
    y_train = y_shuffled[:training_idx]
    X_test = X_shuffled[training_idx:]
    y_test = y_shuffled[training_idx:]


    # Convert data to PyTorch tensors
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64).reshape(-1,1)
    y_test = np.array(y_test, dtype=np.float64).reshape(-1, 1)

    
    ######################################## model to predict loss values from hazard values ########################################

    train_GP = False
    train_neural_network_hazard = False
    train_GBM = True

    a = 2

    inputs_train = X_train[:,a:]
    targets_train = y_train
    inputs_test = X_test[:,a:]
    targets_test = y_test

    plot(X_train[:,0], X_train[:,1], targets_train, np.min(targets_train), np.max(targets_train), "AAL_GRs training")
    plot(X_train[:,0], X_train[:,1], inputs_train[:,0], np.min(inputs_train[:,0]), np.max(inputs_train[:,0]), "AAL_GUs training")

    if train_GP:

        lengthscale = 1
        variance = 10

        kernel = GPy.kern.RBF(input_dim=13, variance=variance, lengthscale=lengthscale)
        model = GPy.models.GPRegression(inputs_train, targets_train, kernel)

        print("Optimizing the model...")
        # Optimize the model
        model.optimize(messages=True)

        targets_pred, y_pred_var = model.predict(inputs_test)

        Error = np.linalg.norm(targets_pred - targets_test)/np.linalg.norm(y_test) *100

        print("Error on single location GR prediction:", Error)

        c = inputs_test[:,-1]
        vmin = c.min()
        vmax = c.max()

        plot(inputs_test[:,0], targets_pred, c, vmin, vmax, "Predictions GU vs GL")
        plot(inputs_test[:,0], targets_test, c, vmin, vmax, "Ground Truth GU vs GL")

        targets_train_pred, y_pred_var = model.predict(inputs_train)

        c = inputs_train[:,-1]
        vmin = c.min()
        vmax = c.max()

        #plot(inputs_train[:,0], targets_train_pred, c, vmin, vmax, "Predictions GU vs GL training")
        #plot(inputs_train[:,0], targets_train, c, vmin, vmax, "Ground Truth GU vs GL training")

    elif train_GBM:

        # Prepare the data for XGBoost
        dtrain = xgb.DMatrix(inputs_train, label=targets_train)
        dtest = xgb.DMatrix(inputs_test, label=targets_test)

        # Set parameters for XGBoost
        params = {
            'max_depth': 5,
            'eta': 0.1,
            'objective': 'reg:squarederror'
        }
        num_round = 100

        print("Training the XGBoost model...")
        # Train the model
        model = xgb.train(params, dtrain, num_round)

        # Make predictions
        targets_pred = model.predict(dtest)

        # Calculate error
        Error = np.linalg.norm(targets_pred.reshape(-1) - targets_test.reshape(-1)) / np.linalg.norm(targets_test.reshape(-1)) * 100

        print("Error from periods prediction:", Error)

        c = inputs_test[:, -1]
        vmin = c.min()
        vmax = c.max()

        xlim = [np.min(inputs_test[:,0]), np.max(inputs_test[:,0])]
        ylim = [np.min(targets_test), np.max(targets_test)]

        plot_v2(inputs_test[:, 0], targets_pred, c, vmin, vmax, "Predictions GU vs GL", xlim, ylim)
        plot_v2(inputs_test[:, 0], targets_test, c, vmin, vmax, "Ground Truth GU vs GL", xlim, ylim)

        c = targets_test
        vmin = c.min()
        vmax = c.max()

        plot(X_test[:, 0], X_test[:, 1], targets_pred, vmin, vmax, "Predictions GL")
        plot(X_test[:, 0], X_test[:, 1], c, vmin, vmax, "Ground Truth GL")

        targets_train_pred = model.predict(dtrain)

        c = inputs_train[:, -1]
        vmin = c.min()
        vmax = c.max()

        # plot(inputs_train[:, 0], targets_train_pred, c, vmin, vmax, "Predictions GU vs GL training")
        # plot(inputs_train[:, 0], targets_train, c, vmin, vmax, "Ground Truth GU vs GL training")


    elif train_neural_network_hazard:

        # Define the neural network architecture
        model = Sequential()
        model.add(Dense(2048, input_dim=5, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        if np.any(np.isnan(inputs_train)) or np.any(np.isnan(targets_train)):
            raise ValueError("Training data contains NaNs")
        if np.any(np.isinf(inputs_train)) or np.any(np.isinf(targets_train)):
            raise ValueError("Training data contains infinite values")
        if np.any(np.isnan(inputs_test)) or np.any(np.isnan(targets_test)):
            raise ValueError("Test data contains NaNs")
        if np.any(np.isinf(inputs_test)) or np.any(np.isinf(targets_test)):
            raise ValueError("Test data contains infinite values")

        # Train the model
        model.fit(inputs_train, targets_train, epochs=100, batch_size=32, validation_data=(inputs_test, targets_test))

        #Error = np.linalg.norm(model.predict(inputs_test) - targets_test)/np.linalg.norm(targets_test) *100

        #print("Error from prediction:", Error)

        c = inputs_test[:,-1]
        vmin = c.min()
        vmax = c.max()

        plot(inputs_test[:,0], model.predict(inputs_test), c, vmin, vmax, "Predictions GU vs GL")
        plot(inputs_test[:,0], targets_test, c, vmin, vmax, "Ground Truth GU vs GL")


    testing_accounts = accounts + 9000

    targets_test = []
    account_ids = []

    for account in range(accounts+1, testing_accounts+1):

        account_id, target = get_account_losses(account)

        targets_test.append(target)
        account_ids.append(account_id)

    targets_test = np.array(targets_test, dtype=float)

    start = account_indexes[accounts+1][0]
    stop = account_indexes[testing_accounts][1]

    results = results_testing[start:stop]

    params = []
    inputs = []
    targets = []
    hazards = []
    num_of_locations = []

    pool = mp.Pool(mp.cpu_count())
    processed_results = list(tqdm(pool.imap(process_result, results), total=len(results)))

    params, inputs, targets, hazards, num_of_locations = zip(*processed_results)
        
    params = np.array(params, dtype=float)
    inputs = np.array(inputs, dtype=float)
    targets = np.array(targets,dtype=float)
    hazards = np.array(hazards,dtype=float)
    num_of_locations = np.array(num_of_locations, dtype=float)

    xs = params[:,0]
    ys = params[:,1]
    values = params[:,2]
    bdeds = params[:,3]
    blims = params[:,4]
    targets = targets[:]
    inputs = inputs[:]

    # Prepare the data for training
    X = np.column_stack((xs, ys, hazards, bdeds, blims, values, num_of_locations))

    targets_pred = np.zeros(targets_test.shape)

    for k, account in enumerate(np.arange(accounts+1, testing_accounts+1)):

        x0 = account_indexes[account][0] - start
        x1 = account_indexes[account][1] - start

        input = X[x0:x1,2:]
        target = targets[x0:x1]

        dtest = xgb.DMatrix(input, label=target)
        t0 = time.time()
        result = model.predict(dtest)
        t1 = time.time()
        if k==200: print("Time to predict:", (t1-t0)*1e6, "microseconds for an account of size:", x1-x0)


        Error = np.linalg.norm(result.reshape(-1) - target.reshape(-1)) / np.linalg.norm(target.reshape(-1)) * 100

        targets_pred[k] = np.sum(result)

        #print("Error", Error)


    Error = np.linalg.norm(targets_pred.reshape(-1) - targets_test.reshape(-1)) / np.linalg.norm(targets_test.reshape(-1)) * 100

    print("Error on GR loss of entire (multiple) location account:", Error)

    

if __name__ == '__main__':
    main()
