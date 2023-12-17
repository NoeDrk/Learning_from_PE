import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import ast
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns



input_file = "/Users/noederijck/Desktop/word_lists/CLEAN_DATA_HOMOSUB/16000normscale_entropyentropy_PE_2.csv"

df = pd.read_csv(input_file)
strategies = ["max", "avg", "rand"]
list_colors = ["#ffd700", "#ffb14e", "#fa8775", "#ea5f94", "#ca0068", "#e00068", "#cd34b5", "#9d02d7", "#0000ff", "#0000af"]


#print(df.columns)

def individual_plots_acc(data, mean_single):
    strategy_info = {
        "avg": {"label": "Average", "color": "#00B050"},
        "min": {"label": "Minimum", "color": "#7030A0"},
        "max": {"label": "Maximum", "color": "#FF0000"},
        "rand": {"label": "Random", "color": "#FFC000"},
    }

    for strat in strategy_info:
        plt.figure(figsize=(12.8, 9.6))
        plt.rcParams.update({'font.size': 24})
        info = strategy_info[strat]
        label, colors = info["label"], info["color"]
        if mean_single == "mean":
           #plt.fill_between(range(len(data[f"mean_model_acc_data_{strat}"])), data[f"mean_model_acc_data_{strat}"] - data[f"SD_model_acc_data_{strat}"], data[f"mean_model_acc_data_{strat}"] + data[f"SD_model_acc_data_{strat}"], alpha=0.2, color=colors)
           pass
        plt.ylim(0, 1)
        plt.xlabel('Epochs', fontsize=32)
        plt.ylabel("Accuracy", fontsize=32)
        plt.plot(data[f"{mean_single}_model_acc_data_{strat}"], label=label, color=colors)
        plt.legend(loc='upper left', fontsize='small')          
        #nipy_spectral
        x = np.arange(len(data[f"{mean_single}_model_acc_data_{strat}"]))
        cmap = plt.cm.get_cmap('Greys', 10)
        norm = mcolors.Normalize(vmin=0, vmax=9)
        
        for i, value in enumerate(data[f"{mean_single}_model_choice_acc_data_{strat}"]):
            color = cmap(1-norm(value))
            plt.plot([x[i], x[i] + 1], [0.02, 0.025], color=color, linewidth=10)
                    
            
        cbar_ax = plt.axes([0.92, 0.3, 0.02, 0.4])  # Adjust the position and size of the colorbar
        cmap_bar = plt.cm.get_cmap('Greys', 10).reversed()
        norm = Normalize(vmin=0, vmax=9)
        sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
        
        sm.set_array([])
        
        # Plot the color gradient on the colorbar
        cbar = plt.colorbar(sm, cax=cbar_ax)
        ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cbar.set_ticks(ticks)
        plt.show()


    
    
def plots_acc(data, mean_single):
    plt.figure(figsize=(12.8, 9.6))
    plt.rcParams.update({'font.size': 24})
    if mean_single == "mean":
        for strat in strategies:
            if strat == "avg":
                colors = "#00B050"
            if strat == "min":
                colors = "#7030A0"
            if strat == "max":
                colors = "#FF0000"
            if strat == "rand":
                colors = "#FFC000"
            plt.fill_between(range(len(data[f"mean_model_acc_data_{strat}"])), data[f"mean_model_acc_data_{strat}"] - data[f"SD_model_acc_data_{strat}"], data[f"mean_model_acc_data_{strat}"] + data[f"SD_model_acc_data_{strat}"], alpha=0.2, color = colors)
    for strat in strategies:
        if strat == "avg":
            colors = "#00B050"
            label = 'Average'
        if strat == "min":
            colors = "#7030A0"
            label = 'Minimum'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum'
        if strat == "rand":
            colors = "#FFC000"
            label = 'Random'
        plt.ylim(0,1)
        plt.xlabel('Epochs', fontsize= 32)
        plt.ylabel("Accuracy", fontsize= 32)
        plt.plot(data[f"{mean_single}_model_acc_data_{strat}"], label = label, color = colors)
    plt.legend(loc='upper left', fontsize='small')
    plt.show()


def plots_entropy(data, mean_single):
    plt.figure(figsize=(12.8, 9.6))
    plt.rcParams.update({'font.size': 24})
    if mean_single == "mean":
        for strat in strategies:
            if strat == "avg":
                colors = "#00B050"
            if strat == "min":
                colors = "#7030A0"
            if strat == "max":
                colors = "#FF0000"
            if strat == "rand":
                colors = "#FFC000"
            #plt.fill_between(range(len(data[f"mean_model_acc_data_{strat}"])), data[f"mean_model_acc_data_{strat}"] - data[f"SD_model_acc_data_{strat}"], data[f"mean_model_acc_data_{strat}"] + data[f"SD_model_acc_data_{strat}"], alpha=0.2, color = colors)
    for strat in strategies:
        if strat == "avg":
            colors = "#00B050"
            label = 'Average'
        if strat == "min":
            colors = "#7030A0"
            label = 'Minimum'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum'
        if strat == "rand":
            colors = "#FFC000"
            label = 'Random'
        plt.ylim(0,3.4)
        plt.xlabel('Epochs', fontsize= 32)
        plt.ylabel("Entropy", fontsize= 32)
        plt.plot(data[f"{mean_single}_model_entropy_data_{strat}"], label = label, color = colors)
    plt.legend(loc='upper left', fontsize='small')
    plt.show()
    

def plots_entropy_accuracy(data, mean_single):
    plt.figure(figsize=(12.8, 9.6))
    plt.rcParams.update({'font.size': 24})
    ax1 = plt.gca()
    strategies = ["avg"]
    if mean_single == "mean":
        for strat in strategies:
            if strat == "avg":
                colors = "#00B050"
            if strat == "min":
                colors = "#7030A0"
            if strat == "max":
                colors = "#FF0000"
            if strat == "rand":
                colors = "#FFC000"
            plt.fill_between(range(len(data[f"mean_model_acc_data_{strat}"])), data[f"mean_model_acc_data_{strat}"] - data[f"SD_model_acc_data_{strat}"], data[f"mean_model_acc_data_{strat}"] + data[f"SD_model_acc_data_{strat}"], alpha=0.2, color = colors)
            plt.fill_between(range(len(data[f"mean_model_entropy_data_{strat}"])), data[f"mean_model_entropy_data_{strat}"] - data[f"SD_model_entropy_data_{strat}"], data[f"mean_model_entropy_data_{strat}"] + data[f"SD_model_entropy_data_{strat}"], alpha=0.2, color = colors)
    
    for strat in strategies:
        if strat == "avg":
            colors = "#00B050"
            label = 'Entropy'
        if strat == "min":
            colors = "#7030A0"
            label = 'Minimum'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum'
        if strat == "rand":
            colors = "#FFC000"
            label = 'Random'
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Epochs', fontsize=32)
        ax1.set_ylabel("Entropy", fontsize=32)
        ax1.plot(data[f"{mean_single}_model_entropy_data_{strat}"], label=label, color=colors)

    ax2 = ax1.twinx()  # Create a twin y-axis
    for strat in strategies:
        if strat == "avg":
            colors = "#008000"
            label = 'Accuracy'
        if strat == "min":
            colors = "#800080"
            label = 'Minimum Acc'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum Acc'
        if strat == "rand":
            colors = "#FFA500"
            label = 'Random Acc'
        ax2.set_ylim(0, 1.0)  # Set appropriate limits for accuracy
        ax2.set_ylabel("Accuracy", fontsize=32)
        ax2.plot(data[f"{mean_single}_model_acc_data_{strat}"], label=label, color=colors, linestyle='--')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize='small')

    plt.show()
    
    
    
def plots_euclid_dist(data, mean_single):
    plt.figure(figsize=(12.8, 9.6))
    plt.rcParams.update({'font.size': 24})
    if mean_single == "mean":
        for strat in strategies:
            if strat == "avg":
                colors = "#00B050"
            if strat == "min":
                colors = "#7030A0"
            if strat == "max":
                colors = "#FF0000"
            if strat == "rand":
                colors = "#FFC000"
            #plt.fill_between(range(len(data[f"mean_model_acc_data_{strat}"])), data[f"mean_model_acc_data_{strat}"] - data[f"SD_model_acc_data_{strat}"], data[f"mean_model_acc_data_{strat}"] + data[f"SD_model_acc_data_{strat}"], alpha=0.2, color = colors)
    for strat in strategies:
        if strat == "avg":
            colors = "#00B050"
            label = 'Average'
        if strat == "min":
            colors = "#7030A0"
            label = 'Minimum'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum'
        if strat == "rand":
            colors = "#FFC000"
            label = 'Random'
        plt.ylim(0,0.3)
        plt.xlabel('Epochs', fontsize= 32)
        plt.ylabel("Changes in weights", fontsize= 32)
        plt.plot(data[f"{mean_single}_model_euclid_dist_{strat}"], label = label, color = colors)
    plt.legend(loc='upper left', fontsize='small')
    plt.show()
    



def plots_sub(data, mean_single):
    subsets = {}
    for strat in strategies:
        plt.figure(figsize=(12.8, 9.6))
        plt.rcParams.update({'font.size': 24})
        for value in range(len(ast.literal_eval(data[f"{mean_single}_subset_data_max"][0]))):
            subsets[value] = []
        for row in range(len(data[f"{mean_single}_subset_data_max"])):
            values = ast.literal_eval(data[f"{mean_single}_subset_data_{strat}"][row])
            for index, value in enumerate(values):
                subsets[index].append(value)
        for key in range(len(subsets)):
            plt.plot(subsets[key], color=list_colors[key], label = f"Subset {key+1}")
            plt.ylim(0,1)
        if strat == "avg":
            s = 'Average'
        if strat == "min":
            s = 'Minimum'
        if strat == "max":
            s = 'Maximum'
        if strat == "rand":
            s = 'Random'
        plt.xlabel('Epochs', fontsize= 32)
        plt.ylabel("Accuracy", fontsize= 32)
        plt.title(f"{s} Strategy")
        plt.legend(fontsize='small', bbox_to_anchor=(1, 1))
        plt.show()



def plots_sub_deriv(data, mean_single):
    subsets = {}
    for strat in strategies:
        chosen_deriv_values = [0]
        agent_choice = data[f"{mean_single}_model_choice_data_{strat}"]
        plt.figure(figsize=(12.8, 9.6))
        plt.rcParams.update({'font.size': 24})
        for value in range(len(ast.literal_eval(data[f"{mean_single}_subset_data_max"][0]))):
            subsets[value] = []
        for row in range(len(data[f"{mean_single}_subset_data_max"])):
            values = ast.literal_eval(data[f"{mean_single}_subset_deriv_data_{strat}"][row])
            for index, value in enumerate(values):
                subsets[index].append(value)
        for i in range(len(agent_choice) - 1):
            for key in range(len(subsets)):
                if agent_choice[i] == key:
                    plt.scatter([i, i+1], [subsets[key][i], subsets[key][i+1]], color=list_colors[key], s=10)
                    plt.ylim(0,0.01)
            #plt.plot(subsets[key], color=list_colors[key], label = f"Subset {key+1}")
            #plt.ylim(0,0.3)
        if strat == "avg":
            colors = "#00B050"
            label = 'Average'
        if strat == "min":
            colors = "#7030A0"
            label = 'Minimum'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum'
        if strat == "rand":
            colors = "#FFC000"
            label = 'Random'
        plt.xlabel('Epochs', fontsize= 32)
        plt.ylabel("Estimate 1st Derivative", fontsize= 32)
        plt.title(f"{label} Strategy")
        #plt.legend(fontsize='small', bbox_to_anchor=(1, 1))
        plt.show()


def plots_sub_sec_deriv(data, mean_single):
    subsets = {}
    for strat in strategies:
        chosen_deriv_values = [0]
        agent_choice = data[f"{mean_single}_model_choice_data_{strat}"]
        plt.figure(figsize=(12.8, 9.6))
        plt.rcParams.update({'font.size': 24})
        for value in range(len(ast.literal_eval(data[f"{mean_single}_subset_data_max"][0]))):
            subsets[value] = []
        for row in range(len(data[f"{mean_single}_subset_data_max"])):
            values = ast.literal_eval(data[f"{mean_single}_subset_sec_deriv_data_{strat}"][row])
            for index, value in enumerate(values):
                subsets[index].append(value)
        for i in range(len(agent_choice) - 1):
            for key in range(len(subsets)):
                if agent_choice[i] == key:
                    plt.scatter([i, i+1], [subsets[key][i], subsets[key][i+1]], color=list_colors[key], s=10)
                    plt.ylim(-0.01,0.01)
       
            #plt.plot(subsets[key], color=list_colors[key], label = f"Subset {key+1}")
            #plt.ylim(0,0.3)
        if strat == "avg":
            colors = "#00B050"
            label = 'Average'
        if strat == "min":
            colors = "#7030A0"
            label = 'Minimum'
        if strat == "max":
            colors = "#FF0000"
            label = 'Maximum'
        if strat == "rand":
            colors = "#FFC000"
            label = 'Random'
        plt.xlabel('Epochs', fontsize= 32)
        plt.ylabel("Estimate 1st Derivative", fontsize= 32)
        plt.title(f"{label} Strategy")
        #plt.legend(fontsize='small', bbox_to_anchor=(1, 1))
        plt.show()

def plots_individual_sub_deriv(data, mean_single):
    subsets = {}
    for strat in strategies:
        chosen_deriv_values = [0]
        agent_choice = data[f"{mean_single}_model_choice_data_{strat}"]
        for value in range(len(ast.literal_eval(data[f"{mean_single}_subset_data_max"][0]))):
            subsets[value] = []
        for row in range(len(data[f"{mean_single}_subset_data_max"])):
            values = ast.literal_eval(data[f"{mean_single}_subset_deriv_data_{strat}"][row])
            for index, value in enumerate(values):
                subsets[index].append(value)
        for key in range(len(subsets)):
            plt.figure(figsize=(12.8, 9.6))
            plt.rcParams.update({'font.size': 24})
            for i in range(len(agent_choice) - 1):
                if agent_choice[i] != key:
                    color = "black"
                else:
                    color = list_colors[key]
                plt.plot([i, i+1], [subsets[key][i], subsets[key][i+1]], color=color)
                plt.ylim(0,0.3)
       
            #plt.plot(subsets[key], color=list_colors[key], label = f"Subset {key+1}")
            #plt.ylim(0,0.3)
            if strat == "avg":
                colors = "#00B050"
                label = 'Average'
            if strat == "min":
                colors = "#7030A0"
                label = 'Minimum'
            if strat == "max":
                colors = "#FF0000"
                label = 'Maximum'
            if strat == "rand":
                colors = "#FFC000"
                label = 'Random'
            plt.xlabel('Epochs', fontsize= 32)
            plt.ylabel("Estimate 1st Derivative", fontsize= 32)
            plt.title(f"Subset {key+1} {label} Strategy")
            #plt.legend(fontsize='small', bbox_to_anchor=(1, 1))
            plt.show()
        
        
def individual_plots_choice_acc(data, mean_single):
    strategy_info = {
        "avg": {"label": "Average", "color": "#00B050"},
        #"min": {"label": "Minimum", "color": "#7030A0"},
        "max": {"label": "Maximum", "color": "#FF0000"},
        "rand": {"label": "Random", "color": "#FFC000"},
    }
    for strat in strategy_info:
        plt.figure(figsize=(12.8, 9.6))
        plt.rcParams.update({'font.size': 24})
        info = strategy_info[strat]
        label, colors = info["label"], info["color"]
        mean_choice_acc = data[f"{mean_single}_model_choice_acc_data_{strat}"]
        plt.hist(mean_choice_acc, bins=10, color= colors, edgecolor='black')
        plt.xlim(0,9)
        plt.xlabel('Mean Distance to Optimal Choice', fontsize= 32)
        plt.ylabel("Count", fontsize= 32)
        plt.title(f"{label} Strategy")
        plt.show()
        
        
def moving_average(data, window_size=25):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def individual_plots_choice_acc_epoch(data, mean_single):
    strategy_info = {
        "avg": {"label": "Average", "color": "#00B050"},
        #"min": {"label": "Minimum", "color": "#7030A0"},
        "max": {"label": "Maximum", "color": "#FF0000"},
        "rand": {"label": "Random", "color": "#FFC000"},
    }

    for strat in strategy_info:
        smoothed_data = moving_average(data[f"{mean_single}_model_choice_acc_data_{strat}"])
        plt.figure(figsize=(12.8, 9.6))
        plt.rcParams.update({'font.size': 24})
        info = strategy_info[strat]
        label, colors = info["label"], info["color"]
        if mean_single == "mean":
           plt.fill_between(range(len(data[f"mean_model_choice_acc_data_{strat}"])), data[f"mean_model_choice_acc_data_{strat}"]  - data[f"SD_model_choice_acc_data_{strat}"], data[f"mean_model_choice_acc_data_{strat}"] + data[f"SD_model_choice_acc_data_{strat}"], alpha=0.2, color=colors)
           #pass
        plt.ylim(0, 9)
        plt.xlabel('Epochs', fontsize=32)
        plt.ylabel("Distance to Optimal Choice", fontsize=32)
        plt.plot(smoothed_data, label=label, color=colors)
        #plt.plot(data[f"{mean_single}_model_choice_acc_data_{strat}"], label=label, color=colors)
        plt.legend(loc='upper left', fontsize='small')          
        plt.show()


def plots_sub_deriv_euclid(data, mean_single):
    subsets = {}
    for strat in strategies:
        fig, ax1 = plt.subplots(figsize=(12.8, 9.6))
        ax2 = ax1.twinx()  # Create a twin Axes for the second y-axis
        plt.rcParams.update({'font.size': 24})

        for value in range(len(ast.literal_eval(data[f"{mean_single}_subset_data_max"][0]))):
            subsets[value] = []

        for row in range(len(data[f"{mean_single}_subset_data_max"])):
            values = ast.literal_eval(data[f"{mean_single}_subset_deriv_data_{strat}"][row])
            for index, value in enumerate(values):
                subsets[index].append(value)

        for key in range(len(subsets)):
            ax1.plot(subsets[key], color=list_colors[key], label=f"Subset {key+1}")
            ax1.set_ylim(0, 0.03)

        if strat == "avg":
            s = 'Average'
        if strat == "min":
            s = 'Minimum'
        if strat == "max":
            s = 'Maximum'
        if strat == "rand":
            s = 'Random'

        ax1.set_xlabel('Epochs', fontsize=32)
        ax1.set_ylabel("1st Order Derivative", fontsize=32)
        ax1.set_title(f"{s} Strategy")

        # Example of plotting another continuous variable on the right y-axis
        # Replace 'your_second_variable_data' with the actual data you want to plot
        your_second_variable_data = data[f"{mean_single}_model_euclid_dist_{strat}"]
        ax2.plot(your_second_variable_data, color='black', label='Euclidean distance')
        ax2.set_ylim(0, 0.2)  # Set the limits for the second y-axis according to your data
        ax2.set_ylabel('Euclidean distance', color='black', fontsize=32)
        ax2.legend()
        plt.show()
        
        
def plots_sub_sec_deriv_euclid(data, mean_single):
    subsets = {}
    for strat in strategies:
        fig, ax1 = plt.subplots(figsize=(12.8, 9.6))
        ax2 = ax1.twinx()  # Create a twin Axes for the second y-axis
        plt.rcParams.update({'font.size': 24})

        for value in range(len(ast.literal_eval(data[f"{mean_single}_subset_data_max"][0]))):
            subsets[value] = []

        for row in range(len(data[f"{mean_single}_subset_data_max"])):
            values = ast.literal_eval(data[f"{mean_single}_subset_sec_deriv_data_{strat}"][row])
            for index, value in enumerate(values):
                subsets[index].append(value)

        for key in range(len(subsets)):
            ax1.plot(subsets[key], color=list_colors[key], label=f"Subset {key+1}")
            ax1.set_ylim(-0.01, 0.01)

        if strat == "avg":
            s = 'Average'
        if strat == "min":
            s = 'Minimum'
        if strat == "max":
            s = 'Maximum'
        if strat == "rand":
            s = 'Random'

        ax1.set_xlabel('Epochs', fontsize=32)
        ax1.set_ylabel("2nd Order Derivative", fontsize=32)
        ax1.set_title(f"{s} Strategy")

        # Example of plotting another continuous variable on the right y-axis
        # Replace 'your_second_variable_data' with the actual data you want to plot
        your_second_variable_data = data[f"{mean_single}_model_euclid_dist_{strat}"]
        ax2.plot(your_second_variable_data, color='black', label='Euclidean distance')
        ax2.set_ylim(0, 0.2)  # Set the limits for the second y-axis according to your data
        ax2.set_ylabel('Euclidean distance', color='black', fontsize=32)
        ax2.legend()
        plt.show()

            
            

#You just have to switch the second input, to either "mean" or the number of a model (0-99)

plots_acc(df, "mean")
plots_entropy(df, "0")
plots_entropy_accuracy(df, "mean")
plots_sub(df, "0")
plots_euclid_dist(df, "mean")
plots_sub_deriv_euclid(df, "mean")
plots_sub_sec_deriv_euclid(df, "mean")
plots_sub_deriv(df, "mean")
plots_sub_sec_deriv(df, "mean")
individual_plots_choice_acc(df, "mean")
individual_plots_choice_acc_epoch(df, "mean")

#PRETTY SHIT GRAPHS
#plots_individual_sub_deriv(df, "mean")
#individual_plots_acc(df, "mean")
