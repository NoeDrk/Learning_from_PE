#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:03:10 2023

@author: noederijck
"""

#SOFTMAX FASHION 6, 0.01


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import math

# The way I store data is awful, I was in too deep, good luck if you need to change it ;)
# It's messy but if you use the "data_cleaning.py" script everything will be ok. 

#PARAMETERS THAT CAN BE MODIFIED
number_images = 6000
num_hid_neuron = 300
num_epochs = 5
num_subsets = 10
steps_to_switch_strat = 1
training_data_size = 1
num_models = 2


#Probability choice strategy
temperature = 0.03
#Learning rate
a = 0.001

log_n = math.log2(num_subsets)


#Only one variable can be set to True, changing which variable is set to True will change which variables are taken into account.
#PE = Prediction Error
#D1 = 1st derivative
#D2 = 2nd derivative
PE = True
PE_D1 = False
PE_D1_D2 = False
D1_D2 = False
PE_D2 = False

for i in range(num_models):
    print(f"{i+1}/{num_models}")
    random_number = np.random.rand()
    random_number2 = np.random.rand()
    
    if PE:
        alphaPE = 1
        name = f"PE_{alphaPE}"
    
    if PE_D1:
        alphaPE = 0.5
        alphaD1 = 0.5
        name = f"PE_D1_{alphaPE}_{alphaD1}"
        
    if PE_D1_D2:
        alphaPE = 1/3
        alphaD1 = 1/3
        alphaD2 = (1 - (alphaPE + alphaD1))
        name = f"PE_D1_D2_{alphaPE}_{alphaD1}_{alphaD2}"
    
    file_name = f"entropy_{name}__{random_number}{random_number2}"
    
    ###Tom Verguts' code
    def preprocess_digits(x_train, y_train, train_size, x_test, y_test, test_size, image_size, n_labels):
        x_train, y_train, x_test, y_test = x_train[:train_size,:], y_train[:train_size], x_test[:test_size,:], y_test[:test_size]
        x_train = x_train.reshape(x_train.shape[0], image_size)  / 255   # from 3D to 2D input data
        x_test  = x_test.reshape(x_test.shape[0], image_size)    / 255   # same here
        y_train = tf.one_hot(y_train, n_labels)
        y_test  = tf.one_hot(y_test, n_labels)
        return x_train, y_train, x_test, y_test	
    
    def norm_array(array):
        if np.min(array) == (np.max(array)):
            max_val = np.max(array) + 0.0000000000001
        else:
            max_val = np.max(array)
        
        norm_array = (array - np.min(array)) / max_val - np.min(array)
        
        return norm_array
    
    #calculates PE for each subset of words and returns the index of the chosen subset (60% chance it chooses it's prefered strategy)
    def choice_subset(subset_train, subset_targets, strat, estimate_subset_derivative, estimate_subset_sec_derivative, entropy):
        # Initialize dictionaries to store the PE and predictions for each subset
        subset_PE_dict = {}
        subset_pred_dic = {}
        # Iterate over each subset to calculate predictions and errors
        for num in range(len(subset_train)):
            # Predict outcomes using the model for the current subset
            subset_pred = model.predict(subset_train[num])
    
            # Calculate the mean absolute error between targets and predictions
            subset_error = np.mean(np.abs(subset_targets[num] - subset_pred))
    
            # Store the absolute error as PE
            subset_PE_dict[num] = abs(subset_error)
    
            # Store the predictions
            subset_pred_dic[num] = subset_pred
    
        # Determine the choice of subset using the strategy function
        choice, index = strategy(PE_dic=subset_PE_dict, strategy=strat, estimate_subset_derivative=estimate_subset_derivative, estimate_subset_sec_derivative=estimate_subset_sec_derivative, entropy=entropy)
    
        # Select a subset based on the calculated choice probabilities
        subset_choice = np.random.choice(len(subset_train), size=1, p=choice)
    
        # Sort the indices based on choice probabilities in descending order
        sorted_indices = np.argsort(-choice)
    
        # Find the position of the chosen subset in the sorted indices
        chosen_start = np.where(sorted_indices == subset_choice)[0][0]
    
        # Return relevant information: the chosen subset, its position, prediction dictionary, PE dictionary, and choice probabilities
        return subset_choice[0], chosen_start, subset_pred_dic, subset_PE_dict, choice
    
    
    def strategy(PE_dic, strategy, estimate_subset_derivative, estimate_subset_sec_derivative, entropy):
        # Convert lists to numpy arrays for efficient computation
        PE_values = np.array(list(PE_dic.values()))
        frst_der_value = np.array(estimate_subset_derivative) + 1
        sec_der_value = np.array(estimate_subset_sec_derivative) + 1
    
        # Normalize the first and second derivative values
        normalized_PE = PE_values  
        normalized_D1 = norm_array(frst_der_value)
        normalized_D2 = norm_array(sec_der_value)
    
        # Include different measures into the agent's choice
        if PE:
            PE_values = normalized_PE
        if PE_D1:
            PE_values = (alphaPE * normalized_PE) + (alphaD1 * normalized_D1)
        if PE_D2:
            PE_values = (alphaPE * normalized_PE) + (alphaD2 * normalized_D2)
        if D1_D2:
            PE_values = (alphaD1 * normalized_D1) + (alphaD2 * normalized_D2)
        if PE_D1_D2:
            PE_values = (alphaPE * normalized_PE) + (alphaD1 * normalized_D1) + (alphaD2 * normalized_D2)
    
        # Ensure entropy is not too low
        entropy = max(entropy, 0.033)
    
        # Strategy decision making
        if strategy == "max":
            # Calculates which subset has the highest PE
            choice = max(zip(PE_dic.values(), PE_dic.keys()))[1]
            max_values = PE_values
            exp_values = np.exp(max_values / temperature)
            P_distribution = exp_values / np.sum(exp_values)
        elif strategy == "min":
            # Calculates which subset has the lowest PE
            choice = min(zip(PE_dic.values(), PE_dic.keys()))[1]
            min_values = -1 * PE_values
            exp_values = np.exp(min_values / temperature)
            P_distribution = exp_values / np.sum(exp_values)
        elif strategy == "avg":
            # Calculates which subset has the most average PE
            average_PE = np.mean(PE_values)
            choice = min(PE_dic.items(), key=lambda x: abs(average_PE - x[1]))[0]
            averaged_PE = -1 * abs(np.array(PE_values) - average_PE * (1/entropy))
            exp_values = np.exp(averaged_PE / temperature)
            P_distribution = exp_values / np.sum(exp_values)
        elif strategy == "rand":
            # Assigns equal probability to each subset
            P_distribution = np.full(len(PE_dic), (1/(len(PE_dic))))
            choice = np.random.choice(list(PE_dic.keys()))
    
        return P_distribution, choice
        
    
        
    # Initialize lists and dictionaries for storing data
    words_list = []
    epoch = 0
    target_class = []
    input_img = []
    list_average_PE = []
    mean_dic_subsets = {}
    
    # Try loading preprocessed data, if not available, preprocess and save
    try:
        # Load data if it has already been processed and saved
        input_img = np.load(f"input_img{number_images}_Fashion.npy")
        target_class = np.load(f"target_class{number_images}_Fashion.npy")
        subset_input_img = np.load(f'subset_input_img{number_images}_Fashion.npy')
        subset_target_class = np.load(f'subset_target_class{number_images}_Fashion.npy')
    except:
        # Load and preprocess data from the Fashion MNIST dataset if not already saved
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (_, _) = fashion_mnist.load_data()
        train_size = number_images  # Downscale to make the dataset smaller and training faster
        image_size = train_images.shape[1] * train_images.shape[2]
        n_labels = int(np.max(train_labels) + 1)
    

        input_img, target_class, x_test, y_test = preprocess_digits(train_images, train_labels, train_size, image_size, n_labels)
        
        # Create subsets of the input images and target classes
        subset_input_img = np.zeros((10, int(number_images / 10), 784))
        subset_target_class = np.zeros((10, int(number_images / 10), 10))
        
        # Distribute images into subsets
        count = [0] * 10  # Counters for each class
        for i in range(len(target_class)):
            index = np.argmax(target_class[i])  # Find the class index
            if count[index] < number_images / 10:
                subset_input_img[index][count[index]] = input_img[i]
                subset_target_class[index][count[index]] = target_class[i]
                count[index] += 1
                
        # Save the processed data for future use
        np.save(f"input_img{number_images}_Fashion.npy", input_img)
        np.save(f"target_class{number_images}_Fashion.npy", target_class)
        np.save(f'subset_input_img{number_images}_Fashion.npy', subset_input_img)
        np.save(f'subset_target_class{number_images}_Fashion.npy', subset_target_class)
    
    # Convert numpy arrays to TensorFlow tensors
    input_img = tf.stack(input_img)
    target_class = tf.stack(target_class)
    target_class_array = tf.identity(target_class)  # Duplicate target_class tensor
    subset_input_img = tf.stack(subset_input_img)
    subset_target_class = tf.stack(subset_target_class)
    
    # Initialize dictionaries to store data
    acc_subsets = {}
    mean_dic_subsets = {}
    mean_list_deriv = {}
    mean_list_sec_deriv = {}
    mean_list_acc = {}
    mean_list_entropy = {}
    mean_list_sub_PE = {}
    mean_list_choices = {}
    mean_list_weight_change = {}
    log_choices = []
    
    # Initialize accuracy list for each subset
    for sub in range(len(subset_target_class)):
        acc_subsets[sub] = []
    
    # Model Parameters
    optimizer = Adam(learning_rate=a)
    
    # Define the Model Structure
    model = Sequential()
    model.add(InputLayer(input_shape=(784,)))  # Input layer for 784 features (28x28 image)
    model.add(Dense(units=num_hid_neuron, activation='relu'))  # Hidden layers
    model.add(Dense(units=num_hid_neuron, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))  # Output layer for 10 classes
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Define strategies for model training
    strategies = ["avg", "rand", "max"]
    
    for s in strategies:
        mean_dic_subsets[s] = {}
        for i in range(num_subsets):
            mean_dic_subsets[s][i] = []
            
    
    for s in strategies:
        mean_list_acc[s] = []
        mean_list_choices[s] = []
        mean_list_sub_PE[s] = []
        mean_list_deriv[s] = []
        mean_list_sec_deriv[s] = []
        mean_list_weight_change[s] = []
        mean_list_entropy[s] = []
    
        

# Iterate over each strategy
    for model_index in strategies:
        # Initialize the Sequential model
        model = Sequential()
        model.add(InputLayer(input_shape=(784,)))  # Input layer for 784 features (28x28 image)
        model.add(Dense(units=num_hid_neuron, activation='relu'))  # Hidden layers with 'relu' activation
        model.add(Dense(units=num_hid_neuron, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))  # Output layer for 10 classes with 'softmax' activation
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
        # Setting the current strategy for the model
        PE_stratergy = model_index
    
        # Initialize lists to store data
        accuracies = []
        entropies = []
        weight_changes = []
        losses = []
        strat_log = []
        choice_log = []
        PEs = []
        list_deriv = []
        list_sec_deriv = []
    
        # Initialize lists for first and second derivative estimates
        lst_estimate_derivative = [0,0,0,0,0,0,0,0,0,0]
        lst_estimate_sec_derivative = [0,0,0,0,0,0,0,0,0,0]
    
        # Initialize training accuracy
        training_accuracy = 0
    
        # Reset accuracy list for each subset
        for sub in range(len(subset_target_class)):
            acc_subsets[sub] = []
    
        # Training loop: continues until the model reaches 30% accuracy
        while training_accuracy < 0.3:
            # Select random indices for training batch
            random_indices = np.random.choice(len(input_img), size=6, replace=False)
    
            # Get training and target data using the selected indices
            training_data = tf.gather(input_img, random_indices)
            target_data = tf.gather(target_class, random_indices)
    
            # Train the model on the selected batch
            history = model.fit(x=training_data, y=target_data, batch_size=30, shuffle=True, verbose=0)
    
            # Make predictions on the entire input data
            prediction = model.predict(input_img)
            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = 0
    
            # Calculate training accuracy
            training_accuracy = np.all(prediction == target_class_array, axis=1).sum() / prediction.shape[0]
    
            # Increment the epoch counter
            epoch += 1
            
        
        last_accuracy = 0
        counter = 0
        epoch = 0
        ### MAIN LOOP ###
        while epoch < num_epochs:
            # Initial setup for the first epoch
            if epoch == 0:
                entropy = 1
        
            # Choosing a subset based on the strategy
            choice, chosen_strat, subset_pred_dic, subset_PE_dict, softmax_p_distrib = choice_subset(
                subset_train=subset_input_img, 
                subset_targets=subset_target_class, 
                strat=PE_stratergy, 
                estimate_subset_derivative=lst_estimate_derivative, 
                estimate_subset_sec_derivative=lst_estimate_sec_derivative, 
                entropy=entropy
            )
        
            # Logging the choices made
            choice_log.append([choice, chosen_strat])
        
            # Calculating normalized PE values and updating entropy
            subset_PE_values = np.array(list(subset_PE_dict.values()))
            subset_normalized_values = subset_PE_values / np.sum(subset_PE_values)
            entropy = (-sum(p * math.log2(p) for p in subset_normalized_values)) / log_n
            entropies.append(entropy)
        
            # Storing Performance Estimations
            PEs.append(subset_PE_dict)
        
            # Calculating and logging accuracy for each subset
            list_avg_accuracy = []
            for i in range(len(subset_pred_dic)):
                # Thresholding predictions to get binary values
                subset_pred_dic[i][subset_pred_dic[i] > 0.5] = 1
                subset_pred_dic[i][subset_pred_dic[i] <= 0.5] = 0
        
                # Calculating accuracy for the subset
                subset_accuracy = np.all(subset_pred_dic[i] == subset_target_class[i], axis=1).sum() / subset_pred_dic[i].shape[0]
                acc_subsets[i].append(subset_accuracy)
                list_avg_accuracy.append(subset_accuracy)
        
            # Training the model on chosen subsets for a number of steps
            for i in range(steps_to_switch_strat):
                # Selecting training data from the chosen subset
                random_indices = np.random.choice(len(subset_input_img[choice]), size=training_data_size, replace=False)
                training_data = tf.gather(subset_input_img[choice], random_indices)
                target_data = tf.gather(subset_target_class[choice], random_indices)
        
                # Getting model weights before and after training to calculate weight changes
                t1_W = model.get_weights()
                history = model.fit(x=training_data, y=target_data, batch_size=training_data_size, shuffle=True, verbose=0)
                t2_W = model.get_weights()
        
                # Calculating Euclidean distance between weight vectors
                euclidean_distance = np.sqrt(np.sum([(np.sum((q - p) ** 2)) for p, q in zip(t1_W, t2_W)]))
        
                # Calculating the first and second derivatives of the PE
                t1_PE = 0 if epoch == 0 else PEs[-2][chosen_strat]
                t2_PE = PEs[-1][chosen_strat]
                estimate_derivative_subset = abs(abs(t1_PE) - abs(t2_PE))
                lst_estimate_derivative[chosen_strat] = estimate_derivative_subset
                list_deriv.append(lst_estimate_derivative.copy())
        
                # Calculating second derivative of the PE
                if len(list_deriv) >= 3:
                    sec_derv = PEs[-1][chosen_strat] - (2 * PEs[-2][chosen_strat]) + PEs[-3][chosen_strat]
                    lst_estimate_sec_derivative[chosen_strat] = sec_derv
                list_sec_deriv.append(lst_estimate_sec_derivative.copy())
        
                # Updating logs for weight changes and accuracies
                weight_changes.append(euclidean_distance)
                accuracies.append(np.mean(list_avg_accuracy))
        
                # Incrementing the epoch counter
                epoch += 1

        for i in acc_subsets:
            mean_dic_subsets[model_index][i].append(acc_subsets[i])
    
        mean_list_choices[model_index].append(choice_log)
        mean_list_entropy[model_index].append(entropies)
        mean_list_acc[model_index].append(accuracies)
        mean_list_weight_change[model_index].append(weight_changes)
        mean_list_sub_PE[model_index].append(PEs)
        mean_list_deriv[model_index].append(list_deriv)
        mean_list_sec_deriv[model_index].append(list_sec_deriv)
        
        
        #with open("/Users/noederijck/Desktop/word_lists/epochs_strat.csv","a") as file:
         #   file.write(f"{model_index}, {len(accuracies)}, {PE_stratergy}, {elapsed_time:.2f}\n")
        
        accuracies = []
        entropies = []
        weight_changes = []
        losses = []
        strat_log = []
        choice_log = []
        PEs = []
        list_deriv = []
    
            

    # Key mappings to rename columns based on different strategies
    key_mapping_sub = {'max': "subset_data_max", 'avg': "subset_data_avg", 'min': "subset_data_min", 'rand': "subset_data_rand"}
    key_mapping_entropy = {'max': "model_entropy_data_max", 'avg': "model_entropy_data_avg", 'min': "model_entropy_data_min", 'rand': "model_entropy_data_rand"}
    key_mapping_acc = {'max': "model_acc_data_max", 'avg': "model_acc_data_avg", 'min': "model_acc_data_min", 'rand': "model_acc_data_rand"}
    key_mapping_weights = {'max': "model_euclid_dist_max", 'avg': "model_euclid_dist_avg", 'min': "model_euclid_dist_min", 'rand': "model_euclid_dist_rand"}
    key_mapping_choice = {'max': "model_choice_data_max", 'avg': "model_choice_data_avg", 'min': "model_choice_data_min", 'rand': "model_choice_data_rand"}
    key_mapping_deriv = {'max': "model_derivative_data_max", 'avg': "model_derivative_data_avg", 'min': "model_derivative_data_min", 'rand': "model_derivative_data_rand"}
    key_mapping_sec_deriv = {'max': "model_sec_deriv_data_max", 'avg': "model_sec_deriv_data_avg", 'min': "model_sec_deriv_data_min", 'rand': "model_sec_deriv_data_rand"}
    
    # Creating new dictionaries with renamed keys based on the strategy
    out_dic_sub = {key_mapping_sub[old_key]: value for old_key, value in mean_dic_subsets.items()}
    out_dic_entropy = {key_mapping_entropy[old_key]: value for old_key, value in mean_list_entropy.items()}
    out_dic_acc = {key_mapping_acc[old_key]: value for old_key, value in mean_list_acc.items()}
    out_dic_weights = {key_mapping_weights[old_key]: value for old_key, value in mean_list_weight_change.items()}
    out_dic_choices = {key_mapping_choice[old_key]: value for old_key, value in mean_list_choices.items()}
    out_dic_deriv = {key_mapping_deriv[old_key]: value for old_key, value in mean_list_deriv.items()}
    out_dic_sec_deriv = {key_mapping_sec_deriv[old_key]: value for old_key, value in mean_list_sec_deriv.items()}
    
    # Converting dictionaries to pandas DataFrames
    subset_df = pd.DataFrame(out_dic_sub)
    entropy_df = pd.DataFrame(out_dic_entropy)
    acc_df = pd.DataFrame(out_dic_acc)
    weights_df = pd.DataFrame(out_dic_weights)
    choice_df = pd.DataFrame(out_dic_choices)
    deriv_df = pd.DataFrame(out_dic_deriv)
    sec_deriv_df = pd.DataFrame(out_dic_sec_deriv)
    
    
    
    for s in strategies:
        misplaced_data = acc_df.loc[0, f'model_acc_data_{s}']
        for i in range(len(misplaced_data)):
            acc_df.loc[i, f'model_acc_data_{s}'] = misplaced_data[i]
            
    for s in strategies:
        misplaced_data = entropy_df.loc[0, f'model_entropy_data_{s}']
        for i in range(len(misplaced_data)):
            entropy_df.loc[i, f'model_entropy_data_{s}'] = misplaced_data[i]
            
    for s in strategies:
        misplaced_data = weights_df.loc[0, f'model_euclid_dist_{s}']
        for i in range(len(misplaced_data)):
            weights_df.loc[i, f'model_euclid_dist_{s}'] = misplaced_data[i]
    
    for s in strategies:
        misplaced_data = choice_df.loc[0, f'model_choice_data_{s}']
        for i in range(len(misplaced_data)):
            choice_df.loc[i, f'model_choice_data_{s}'] = misplaced_data[i]
            
    for s in strategies:
        misplaced_data = deriv_df.loc[0, f'model_derivative_data_{s}']
        for i in range(len(misplaced_data)):
            #print(misplaced_data[i])
            deriv_df.loc[i, f'model_derivative_data_{s}'] = misplaced_data[i]
            
    for s in strategies:
        misplaced_data = sec_deriv_df.loc[0, f'model_sec_deriv_data_{s}']
        for i in range(len(misplaced_data)):
            #print(misplaced_data[i])
            sec_deriv_df.loc[i, f'model_sec_deriv_data_{s}'] = misplaced_data[i]
    
    # Update the DataFrame to fix the placement of the data
    
    final_df =  pd.concat([subset_df, acc_df, choice_df, deriv_df, sec_deriv_df, weights_df, entropy_df], axis=1)
    
    new_df = pd.DataFrame()
    for s in strategies:
        new_col = []
        misplaced_data = final_df[f'subset_data_{s}']
        for num in range(len(misplaced_data[0][0])):
            temp_list = []
            for i in range(num_subsets):
                data = misplaced_data[i][0]
                temp_list.append(data[num])
            new_col.append(temp_list)
            
        new_df[f'subset_data_{s}'] = new_col
        new_df[f"model_entropy_data_{s}"] = final_df[f"model_entropy_data_{s}"]
        new_df[f"model_acc_data_{s}"] = final_df[f"model_acc_data_{s}"]
        new_df[f"model_choice_data_{s}"] = final_df[f"model_choice_data_{s}"]
        new_df[f'subset_deriv_data_{s}'] = final_df[f"model_derivative_data_{s}"]
        new_df[f'subset_sec_deriv_data_{s}'] = final_df[f"model_sec_deriv_data_{s}"]
        new_df[f'model_euclid_dist_{s}'] = final_df[f"model_euclid_dist_{s}"]
        
        
    new_df.to_csv(f'/Users/noederijck/Desktop/word_lists/DATA_HOMOSUBS/{number_images}img1/normscale_entropy{file_name}.csv', index=False)
    print("FINISHED")
