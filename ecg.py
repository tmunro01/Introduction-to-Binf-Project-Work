from dataclasses import field

import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import glob
import os
import neurokit2 as nk
from scipy.signal import firwin, lfilter
from tornado.web import is_absolute

""" This piece of code downloads the ECG data, calculates QT intervals and performs the 
anova tests for different genotype groups (AA, AB, BB). The ECG data analysis, i.e., calculating the
QT intervals is ready-made for you. Finding the associations between different groups is done by you. 
Hint for these are in the code! Read all the comments carefully!!"""

# MODIFY YOUR PATH
path = 'C:/Users/thoma/Documents/TAU_2024_P1/Introduction_to_biomedical_informatics/python_project_folder/intro_to_binf_pycharm_project/python_files/project_work/data'


# This function loads the ECG data files: .dat, and .hea -files. .dat files consists of tne ECG voltage data
# and .hea files consist of the header information (e.g., sampling frequency, leads etc.)
def loadEcg(path):
    dat_files = glob.glob(os.path.join(path, "*.dat"))
    hea_files = glob.glob(os.path.join(path, "*.hea"))
    base_names = set([os.path.splitext(os.path.basename(f))[0] for f in dat_files + hea_files])

    ecg_dict, field_dict, fs_dict = {}, {}, {}

    # Read the signal and metadata for each file. The read file consist of field names (contain, e.g.,
    # the sampling frequency and the lead names), and the actual ecg signal data.
    for i, base_name in enumerate(sorted(base_names), start=1):
        ecg, fields = wfdb.rdsamp(os.path.join(path, base_name))
        patient_key = f'Patient{i}'
        ecg_dict[patient_key] = ecg
        field_dict[patient_key] = fields
        fs_dict[patient_key] = fields['fs']

    print(fs_dict)
    return ecg_dict, field_dict, fs_dict


# Function to filter ECG signal. You can modify the filter, i.e. create your own IIR
# or FIR filter with order and cut-off of your choosing. Justify your selection!
def filterEcg(signal, fs, filter_order=20, cutoff=[0.5, 150]):
    filter_coeffs = firwin(filter_order, cutoff, pass_zero=False, fs=fs)
    # Use Lead II (index 1). You can also create a variable for this, e.g. leadII = signal[:,1].
    return lfilter(filter_coeffs, [1.0], signal[:, 1])


# Function to calculate QT intervals.
    # Process the ECG signal using neurokit2 (detects R-peaks, Q, and T points)
    # ecg_process has many steps:
    # 1) cleaning the ecg with ecg_clean()
    # 2) peak detection with ecg_peaks()
    # 3) HR calculus with signal_rate()
    # 4) signal quality assessment with ecg_quality()
    # 5) QRS elimination with ecg_delineate() and
    # 6) cardiac phase determination with ecg_phase().
def calculateQtIntervals(key, filtered_signal, fs):

    #if key != "Patient10":
        #return

    ecg_analysis, _ = nk.ecg_process(filtered_signal, sampling_rate=fs)
    q_points = ecg_analysis['ECG_Q_Peaks'] # This is default output of the ecg_process.
    t_points = ecg_analysis['ECG_T_Offsets'] # This is default output of the ecg_process.
    q_indices = q_points[q_points == 1].index.to_list()
    t_indices = t_points[t_points == 1].index.to_list()

    time = np.arange(filtered_signal.size) / fs

    # YOU NEED TO UNCOMMENT THE PLOTTING COMMANDS FOR ANALYSIS TASK 2.  NOTE, if you plot all the
    # figures at the same time, you get memory load warning. You can modify this to plot
    # only few.
    #plt.figure(figsize=(10, 6))
    #plt.plot(time, filtered_signal, label='Filtered ECG Lead II')

    # Calculate QT intervals and plot them as red lines
    qt_intervals = []
    for q, t in zip(q_indices, t_indices):
        if t > q:  # Ensure T point is after Q point for a valid QT interval
            qt_interval = (t - q) / fs  # The indexes are in samples, thus we need to convert them to seconds.
            qt_intervals.append(qt_interval)

            # Plot the QT interval as a red line segment. YOU NEED TO RUN ALSO THESE FOR TASK 2.
            #plt.plot([q / fs, t / fs], [filtered_signal[q], filtered_signal[t]], color='red', lw=2,
                     #label='QT Interval' if len(qt_intervals) == 1 else "")  # Label only the first for legend clarity

    # YOU NEED TO RUN ALSO THESE FOR TASK 2.
    #plt.xlabel('Time (s)')
    #plt.ylabel('Amplitude')
    #plt.title(f'{key} - Filtered ECG with QT Intervals with 0.5-150hz bandpass')
    #plt.legend()
    #plt.grid(True)
    #plt.ion()
    #plt.show()

    return qt_intervals


# Function to calculate and store average QT interval
def calculateAverageQt(ecg_dict, fs_dict):
    # The average QT intervals for all patients will be stored in average_qt_dict
    average_qt_dict = {}
    #if key == "Patient :
    for key, ecg_signal in ecg_dict.items():
        fs = fs_dict[key] # corresponding sampling freq. for each signal
        filtered_signal = filterEcg(ecg_signal, fs) # This calls the filtering function
        qt_intervals = calculateQtIntervals(key, filtered_signal, fs) # Calculates the intervals based on the filtered data
        average_qt_interval = np.mean(qt_intervals) if qt_intervals else None # Calculates the average QT for each patient.
        average_qt_dict[key] = average_qt_interval
    return average_qt_dict


# TASK 3: Function to load genotype data and reshape (reshaping to keep the structure for 7 rows). New genotype starts
# every 7th row (if you look at the result txt file, you can see this).

def loadAndReshapeGenotype(filepath):
    results = pd.read_csv(filepath, delimiter="\t", header=None)
    selected = results.iloc[:110, 1::7]  # Select every 7th column starting from index 1 --> each new genotype AA, AB or BB.
    s_array = selected.values
    reshaped = s_array.reshape(7, -1)  # Automatically calculate columns based on data size, when we want to have
                                        # the original 7 rows.
    return reshaped

# TASK 3: Function to extract QT intervals based on genotype AA, AB or BB. Thus, this function goes through the reshaped
# data, consisting of 7 rows and 129 columns. 129 is the number of different genotypes i.e. different patients.
# One group to study = same genotype from one row. E.g. all BB genotypes from row 1. YOU NEED THIS INFORMATION ABOUT
# ALL 7 ROWS / GROUPS.

def QtByGenotype(reshaped, average_qt_dict):
    patients = list(average_qt_dict.keys())

    # Change these to calculate for different groups, i.e. rows (rows 1-7, but python indexed 0-6). You can also
    # make this more automated, to loop over each rows in one piece of code.

    # Loop over every row of which there are 7
    for row in range(7):
        AB = np.where(reshaped[row, :] == "AB")[0] # This is the third row (remember, that python indexing starts at 0)
        # Check for AB in row
        if len(AB) == 0:
            is_AB = 0
        else:
            is_AB = 1


        BB = np.where(reshaped[row, :] == "BB")[0]
        # Check for BB in row
        if len(BB) == 0:
            is_BB = 0
        else:
            is_BB = 1

        AA = np.where(reshaped[row, :] == "AA")[0]
        # Check for AA in row
        if len(AA) == 0:
            is_AA = 0
        else:
            is_AA = 1

        qt_AB = [average_qt_dict[patients[idx]] for idx in AB if average_qt_dict[patients[idx]] is not None]
        qt_BB = [average_qt_dict[patients[idx]] for idx in BB if average_qt_dict[patients[idx]] is not None]
        qt_AA = [average_qt_dict[patients[idx]] for idx in AA if average_qt_dict[patients[idx]] is not None]

        # Check if ANOVA, t-test or no test at all will be performed
        test_chosen = is_AB + is_BB + is_AA

        genotypes_present = []
        # Append to genotypes_present if value is 1 ie. they are truthy
        if is_AB:
            genotypes_present.append(qt_AB)
        if is_BB:
            genotypes_present.append(qt_BB)
        if is_AA:
            genotypes_present.append(qt_AA)

        # If only 1 genotype present: no test and inform user
        if test_chosen == 1:
            print(f"{row + 1}. Only 1 genotype present: AB = {is_AB} BB = {is_BB} AA = {is_AA}. No test can be performed")

        # If 2 genotypes present: perform t-test
        if test_chosen == 2:
            print(f"{row + 1}. 2 genotypes present: AB = {is_AB} BB = {is_BB} AA = {is_AA}. Performing t-test : {f_oneway(genotypes_present[0], genotypes_present[1])}.")

        # If 3 genotypes present: perform ANOVA
        if test_chosen == 3:
            print(f"{row + 1}. 3 genotypes present. Performing ANOVA: {f_oneway(qt_AB, qt_BB, qt_AA)}")


# Main processing function
def main():
    ecg_dict, field_dict, fs_dict = loadEcg(path)
    average_qt_dict = calculateAverageQt(ecg_dict, fs_dict)
    #print(average_qt_dict)
    #print(ecg_dict)

    # LOAD HERE YOUR GENOTYPE DATA FROM THE RESULTS YOU GOT IN DATA ANALYSIS TASK 1. Use the
    # loadAndReshapeGenotype function, and assign the result to a certain variable.
    gt_data = loadAndReshapeGenotype("C:/Users/thoma/Documents/TAU_2024_P1/Introduction_to_biomedical_informatics/python_project_folder/intro_to_binf_pycharm_project/python_files/project_work/gt_data_filtered_final.txt")
    #print(gt_data)
    #print(len(gt_data))
    #for patients in gt_data:
        #print(len(patients))

    # Conduct ANOVA or T-test depending on number of genotypes present
    QtByGenotype(gt_data, average_qt_dict)


# Run the main function
if __name__ == "__main__":
    main()
