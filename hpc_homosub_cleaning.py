#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:38:59 2023

@author: noederijck
"""

import os
import pandas as pd
import ast
import numpy as np

rdm = np.random.randint(0,1)

temp = 1
data_set = "fashion"

file_path = "/Users/noederijck/Desktop/word_lists/DATA_HOMOSUBS/6000img1/"
clean_model_file_path = "/Users/noederijck/Desktop/word_lists/CLEAN_DATA_HOMOSUB/"

final_df = pd.DataFrame()
files = os.listdir(file_path)
try:
    files.remove(".DS_Store")
except:
    pass
number_mods = len(files)
name_file = files[0].split("_")[:5]

num_training = name_file[2].split("OUTPUT")[0]

name_file = name_file[0] +"_"+ name_file[1] +"_"+ str(num_training) + "_" + str(number_mods) + ".csv"


columns = ["subset_data", "subset_deriv_data", "model_acc_data", "model_choice_data", "model_choice_acc_data", "subset_sec_deriv_data", "model_euclid_dist", "model_entropy_data"]
#columns = ["subset_data", "subset_deriv_data", "model_acc_data", "model_choice_data", "model_choice_acc_data", "subset_sec_deriv_data", "model_euclid_dist"]
strategies = ["max", "rand", "avg"]

output_file_name = f""


for i in range(len(files)):
    some_data = pd.read_csv(f"{file_path}{files[i]}")
    for col in some_data.columns:
        final_df[f"{i}_{col}"] = some_data[col]


for s in strategies:
    for c in columns:
        final_df[f"mean_{c}_{s}"] = 0
        final_df[f"SD_{c}_{s}"] = 0
        final_df[f"mean_{c}_{s}"] = final_df[f"mean_{c}_{s}"].astype(object)
        final_df[f"SD_{c}_{s}"] = final_df[f"SD_{c}_{s}"].astype(object)
        
for model in range(number_mods):
    for s in strategies:
        choice_data = []
        choice_acc_data = []
        for string in final_df[f"{model}_model_choice_data_{s}"]:
            temp_l = ast.literal_eval(string)
            choice_data.append(temp_l[0])
            choice_acc_data.append(temp_l[1])
        final_df[f"{model}_model_choice_data_{s}"] = choice_data
        final_df[f"{model}_model_choice_acc_data_{s}"] = choice_acc_data
        
                
                
    

def calculating_mean_and_se(list_values):
    data = []
    mean_data = []
    se_data = []
    if isinstance(list_values[0], str):
        for string in list_values:
            temp_l = ast.literal_eval(string)
            data.append(temp_l)
        if len(data[0]) == 2:
            print(data)
        # Calculate the mean and standard error for each column (element-wise)
        mean_value = [np.mean(col) for col in zip(*data)]
        se_value = [np.std(col) for col in zip(*data)]
        #print(se_value)
        mean_data.append(mean_value)
        se_data.append(se_value)
    else:
        mean_data = [np.mean(np.array(list_values))]
        se_data = [np.std(np.array(list_values)) / np.sqrt(len(list_values))]
    return mean_data, se_data



        
for row in range(len(final_df)):
    for s in strategies:
        for col in columns:
            temp_values = []
            for mod in range(number_mods):
                temp_values.append(final_df.loc[row, f"{mod}_{col}_{s}"]) 
            mean_col_val, SD_col_val = calculating_mean_and_se(temp_values)
            final_df.at[row, f"mean_{col}_{s}"] = mean_col_val[0]
            final_df.at[row,f"SD_{col}_{s}"] = SD_col_val[0]


        
final_df.to_csv(f'{clean_model_file_path}{output_file_name}16000{name_file}', index=False)

        
print(f'{clean_model_file_path}{output_file_name}16000{name_file}')