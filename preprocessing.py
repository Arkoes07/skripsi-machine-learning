import os
import time
import scipy
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

##
# SOME CONSTANT
##
DATASET_PATH = "D:/datasets/UTA-RLDD/csv/"
FPS_INFO_PATH = "D:/datasets/UTA-RLDD/fps/fps.txt"

MINUTES_LENGTH = 3 
EYE_THRESHOLD = 0.5
MOUTH_THRESHOLD = 0.5

###
# Function to count total of NaN values exist in the DataFrame
# This function will return True if there is less then or equal 10% NaN values and vice versa
###
def isSufficient(df):
    for idx_class, df_class in df.groupby("class"):
        # check only in rEar, because total of NaN values for rEar, lEar, and mar is the same.
        nanValuesCount = df_class.isna().sum().rEar
        # calculate ratio
        ratio = nanValuesCount/len(df_class)
        # check ratio
        if ratio > 0.1:
            # there is more than 10 percent NaN values
            return False
    # there is less than 10 percent NaN values
    return True

###
# Function to change the class label from 0,5,10 into 1,2,3
# This function will return the modified DataFrame
###
def changeClassLabel(df, data=True):
    if df["class"].dtypes == 'object':
        # contains 10_1 and 10_2
        if data:
            # if it is feature data then update index
            last_10_1_idx = df.loc[df["class"] == "10_1"].frame.values[-1]
            df.loc[df["class"] == "10_2","frame"] = df.loc[df["class"] == "10_2"]["frame"] + last_10_1_idx
        val_dict = dict([('0', '0'), ('5', '1'), ('10','2'), ('10_1','2'), ('10_2','2')])
        df['class'].replace(val_dict, inplace=True)
        df['class'] = df['class'].astype('int64')
    else:
        # do not contains 10_1 and 10_2
        val_dict = dict([(0, 0), (5, 1), (10,2), (10_1,2), (10_2,2)])
        df['class'].replace(val_dict, inplace=True)
    return df
    
###
# Function to set max value of eye aspect ratio into Q3 + 1.5*IQR
# This function will return the modified DataFrame
###
def normalizeMaxValue(df):
    # get q3, iqr, and max value
    rEar_q3 = df.describe().rEar['75%']
    rEar_iqr = rEar_q3 - df.describe().rEar['25%']
    max_rEar = rEar_q3 + (1.5 * rEar_iqr)
    lEar_q3 = df.describe().lEar['75%']
    lEar_iqr = lEar_q3 - df.describe().lEar['25%']
    max_lEar = lEar_q3 + (1.5 * lEar_iqr)
    # apply it to the dataframe
    df.loc[df['rEar'] > max_rEar, 'rEar'] = max_rEar
    df.loc[df['lEar'] > max_lEar, 'lEar'] = max_lEar
    return df
    
###
# Funtion to perform max value normalization and min max scaler for transforming data
# This function will return the modified DataFrame
###
def transform(df):
    # set max eye aspect ratio to q3+1.5iqr
    df = normalizeMaxValue(df)
    # perform min max scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    df[['rEar','lEar','mar']] = scaler.fit_transform(df[['rEar','lEar','mar']])
    return df

### 
# Function to calculate PERCLOS on the DataFrame
# This functioin will return the modified DataFrame
### 
def calculate_perclos(df, window_size):
    df['eye_closed'] = 0
    df.loc[(df['rEar'] < EYE_THRESHOLD) & (df['lEar'] < EYE_THRESHOLD), 'eye_closed'] = 1
    start_frame = 1
    for current_idx in range(window_size+1,len(df)+1):
        stop_frame = current_idx - 1
        df.loc[current_idx,'perclos'] = df.loc[start_frame:stop_frame,'eye_closed'].eq(1).sum() / window_size
        start_frame = start_frame + 1  
    df.drop(['eye_closed'], axis=1, inplace=True)
    return df

### 
# Function to track consecutivr eye closed and mouth opened
# This functioin will return the modified DataFrame
### 
def track_consecutive(df):
    # default value
    df['btb_eye_closed'] = 0
    df['btb_mouth_opened'] = 0
    # check if eye closed and mouth open
    df.loc[(df['rEar'] < EYE_THRESHOLD) & (df['lEar'] < EYE_THRESHOLD), 'btb_eye_closed'] = 1
    df.loc[df['mar'] > MOUTH_THRESHOLD, 'btb_mouth_opened'] = 1
    # track consecutive
    eye = df['btb_eye_closed']
    mouth = df['btb_mouth_opened']
    df['btb_eye_closed'] = eye * (eye.groupby((eye != eye.shift()).cumsum()).cumcount() + 1)
    df['btb_mouth_opened'] = mouth * (mouth.groupby((mouth != mouth.shift()).cumsum()).cumcount() + 1)
    return df

### 
# function to compute microsleep and yawning event rate (event/s) occured in some minutes length
# This functioin will return the modified DataFrame
### 
def calculate_event_rate(df, window_size, fps, btb_microsleep, btb_yawning):
    start_frame = 1
    for current_idx in range(window_size+1,len(df)+1):
        stop_frame = current_idx - 1
        df.loc[current_idx,'microsleep_rate'] = df.loc[start_frame:stop_frame,'btb_eye_closed'].eq(btb_microsleep).sum() / (window_size/fps)
        df.loc[current_idx,'yawning_rate'] = df.loc[start_frame:stop_frame,'btb_mouth_opened'].eq(btb_yawning).sum() / (window_size/fps)
        start_frame = start_frame + 1  
    return df

### 
# function that combine 3 above function into one function
# This functioin will return the modified DataFrame
### 
def feature(df, fps, btb_microsleep, btb_yawning):
    window_size = int(fps * 60 * MINUTES_LENGTH)
    df = calculate_perclos(df, window_size)
    df = track_consecutive(df)
    df = calculate_event_rate(df, window_size, fps, btb_microsleep, btb_yawning)
    df.drop(['btb_eye_closed', 'btb_mouth_opened'], axis=1, inplace=True)
    return df

print("\n== PROGRAM START ==\n-- Data Preprocessing By Alwi")
start_time = time.time()

# get fps information for each video
df_fps = pd.read_csv(FPS_INFO_PATH, delimiter=';', names=['subject','class','fps'], index_col=False)
df_fps = df_fps.astype({"class": int})
# change class label from 0,5,10 into 0,1,2
df_fps = changeClassLabel(df_fps, False)

# containers
data = {"class_0":[], "class_1":[], "class_2":[]}

# loop through csv folder
for filename in os.listdir(DATASET_PATH):
    # for each csv files
    if filename.endswith(".csv"):
        print(f"\n-- processing {filename}...")
        # create filepath
        subject = int(filename[:2])
        filepath = DATASET_PATH + filename
        # create dataframe from csv
        df = pd.read_csv(filepath, delimiter=';', names=['subject','class','frame','rEar','lEar','mar'])
        # change class label from 0,5,10 into 0,1,2
        df = changeClassLabel(df)
        # pass or failed the NaN ratio test
        if not isSufficient(df):
            # failed, there is more than 10 percent NaN values, continue to next failed
            print(f"failed : {filename}, there is more than 10% NaN values exists")
            continue
        # fill the NaN value
        df.fillna(method="ffill", inplace=True)
        # transform data
        df = transform(df)
        # loop through each class
        groups = df.groupby("class")
        for idx_class in range(3):
            # DataFrame for each class
            df_class = groups.get_group(idx_class).set_index('frame')
            # get fps for this subject and class
            fps = df_fps.loc[(df_fps["class"] == idx_class) & (df_fps["subject"] == subject)].fps.values[0]
            # number of consecutive frame to consider as microsleep and yawning
            btb_microsleep = int(fps) # 1 second
            btb_yawning = int(fps * 6) # 6 second
            # perform feature engineering
            df_class = feature(df_class, fps, btb_microsleep, btb_yawning)
            # drop NaN rows, which is the first MINUTES_LENGTH minutes
            df_class.dropna(inplace=True)
            # insert into containers, excluding subject number
            key = "class_"+str(idx_class)
            data[key].extend(df_class.iloc[:,1:].values)
            print(f"{len(df_class)} data inserted into class {idx_class} containers")
        print(f"-- succeed: {filename} successfully preprocessed")

# create DataFrame from containers
df_class_0 = pd.DataFrame(data["class_0"], columns=['class','rEar','lEar','mar','perclos','microsleep_rate','yawning_rate'])
df_class_1 = pd.DataFrame(data["class_1"], columns=['class','rEar','lEar','mar','perclos','microsleep_rate','yawning_rate'])
df_class_2 = pd.DataFrame(data["class_2"], columns=['class','rEar','lEar','mar','perclos','microsleep_rate','yawning_rate'])

# write to a file
df_class_0.to_pickle("./preprocessed-data/class_0.pkl")
df_class_1.to_pickle("./preprocessed-data/class_1.pkl")
df_class_2.to_pickle("./preprocessed-data/class_2.pkl")

print(f"\n== PROGRAM FINISHED in {time.time() - start_time} seconds == ")
