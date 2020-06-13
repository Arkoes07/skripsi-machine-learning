import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import random
import scipy
import tensorflow as tf
import time
from collections import deque
from sklearn.preprocessing import MinMaxScaler

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def arrToString(arr):
    text = ""
    for el in arr:
        text = text + '_' + str(el) 
    return text


class Preprocess:

    ### class variables
    eye_threshold = 0.5
    mouth_threshold = 0.5
    btb_microsleep_second = 1
    btb_yawning_second = 6

    ###
    # Function to change class variable from outside
    ###
    def config(eye_threshold, mouth_threshold, btb_microsleep_second, btb_yawning_second):
        Preprocess.eye_threshold = eye_threshold
        Preprocess.mouth_threshold = mouth_threshold
        Preprocess.btb_microsleep_second = btb_microsleep_second
        Preprocess.btb_yawning_second = btb_yawning_second

    ###
    # Function to count total of NaN values exist in the DataFrame
    # This function will return True if there is less then or equal 10% NaN values and vice versa
    ###
    def isSufficient(df, ratio_threshold):
        for idx_class, df_class in df.groupby("class"):
            # check only in rEar, because total of NaN values for rEar, lEar, and mar is the same.
            nanValuesCount = df_class.isna().sum().rEar
            # calculate ratio
            ratio = nanValuesCount/len(df_class)
            # check ratio
            if ratio > ratio_threshold:
                # there is more than 10 percent NaN values
                return False
        # there is less than 10 percent NaN values
        return True

    ### PRIVATE
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
        
    ### PRIVATE
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
        df = Preprocess.normalizeMaxValue(df)
        # perform min max scaling
        scaler = MinMaxScaler(feature_range=(0,1))
        df[['rEar','lEar','mar']] = scaler.fit_transform(df[['rEar','lEar','mar']])
        return df
    
    ###
    # Function to change fps smaller 
    # return adjusted df
    ###
    def change_fps(df, fps, to_fps):
        if to_fps < fps:
            step = int(fps/to_fps)
            df_step = df.iloc[::step]
            df_step.index = pd.RangeIndex(1, len(df_step)+1, 1)
            return pd.DataFrame(df_step)
        else:
            return df

    ### PRIVATE
    # Function to calculate PERCLOS on the DataFrame
    # This functioin will return the modified DataFrame
    ### 
    def calculate_perclos(df, window_size):
        df['eye_closed'] = 0
        df.loc[(df['rEar'] < Preprocess.eye_threshold) & (df['lEar'] < Preprocess.eye_threshold), 'eye_closed'] = 1
        start_frame = 1
        for current_idx in range(window_size+1,len(df)+1):
            stop_frame = current_idx - 1
            df.loc[current_idx,'perclos'] = df.loc[start_frame:stop_frame,'eye_closed'].eq(1).sum() / window_size
            start_frame = start_frame + 1  
        df.drop(['eye_closed'], axis=1, inplace=True)
        return df

    ### PRIVATE
    # Function to track consecutivr eye closed and mouth opened
    # This functioin will return the modified DataFrame
    ### 
    def track_consecutive(df):
        # default value
        df['btb_eye_closed'] = 0
        df['btb_mouth_opened'] = 0
        # check if eye closed and mouth open
        df.loc[(df['rEar'] < Preprocess.eye_threshold) & (df['lEar'] < Preprocess.eye_threshold), 'btb_eye_closed'] = 1
        df.loc[df['mar'] > Preprocess.mouth_threshold, 'btb_mouth_opened'] = 1
        # track consecutive
        eye = df['btb_eye_closed']
        mouth = df['btb_mouth_opened']
        df['btb_eye_closed'] = eye * (eye.groupby((eye != eye.shift()).cumsum()).cumcount() + 1)
        df['btb_mouth_opened'] = mouth * (mouth.groupby((mouth != mouth.shift()).cumsum()).cumcount() + 1)
        return df

    ### PRIVATE
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
    def feature(df, fps, minutes_length):
        window_size = int(fps * 60 * minutes_length)
        df = Preprocess.calculate_perclos(df, window_size)
        df = Preprocess.track_consecutive(df)
        btb_microsleep = Preprocess.btb_microsleep_second * fps
        btb_yawning = Preprocess.btb_yawning_second * fps
        df = Preprocess.calculate_event_rate(df, window_size, fps, btb_microsleep, btb_yawning)
        df.drop(['btb_eye_closed', 'btb_mouth_opened'], axis=1, inplace=True)
        df.dropna(inplace=True)
        return df
    
    ###
    # function to make a array of fixed sequential window of dataframe
    # return list
    ###
    def sequencialize(df, seq_len):
        sequential_data = []
        prev = deque(maxlen=seq_len)
        for row in df.values:
            prev.append([data for data in row[2:]])
            if len(prev) == seq_len:
                sequential_data.append(np.array(prev))
        return sequential_data
    
    ###
    # function that takes data container and balanced length of each class tham labelling each row
    # return balanced and labelled array
    ###
    def balancing_labelling(arr):
        # balancing
        length_arr = []
        for group in arr:
            length_arr.append(len(group))
        lower = min(length_arr)
        balanced = []
        for group in arr:
            balanced.append(group[:lower])
        # labeling
        data_with_label = [[row,class_idx] for class_idx in range(3) for row in balanced[class_idx]]
        return data_with_label

    
class Preparation:
    
    ###
    # This function will shuffle the given data and split into X and 
    # return two array X and y
    ###
    def shuffle_split_xy(data):
        ## Shuffle data order
        random.shuffle(data)
        ## Split
        X = []
        y = []
        for d_in, d_out in data:
            X.append(d_in)
            y.append(d_out)
        return X,y
    
    ### 
    # This function will open the path and combine all the data
    # return 4 array X_train, X_test, y_train, y_test
    ###
    def data_xy(path, fold, total_fold): 
        # get all filenames in folder path
        filenames = [name for name in os.listdir(path) if name.endswith('.npy')]
        # get test portion
        test_window = int(len(filenames) / total_fold)
        test_idx = range(fold * test_window, fold * test_window + test_window)
        # train test container
        train = []
        test = []
        for idx, filename in enumerate(filenames,start=0):
            # read data
            data = np.load(path+filename, allow_pickle=True)
            # put into container accordingly
            if idx in test_idx:
                test.extend(data)
            else:
                train.extend(data)
        # shuffle and split x y
        X_train, y_train = Preparation.shuffle_split_xy(train)
        X_test, y_test = Preparation.shuffle_split_xy(test)
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    
class ModelContainer:
    
    logs_path = "comparison-logs\\"
    
    def config_train_logs_path(logs_path):
        ModelContainer.logs_path = logs_path
    
    def FCModel(X, X_t, y, y_t, layers, epochs, model_name):
        # create callback
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"{ModelContainer.logs_path}{model_name}-l{arrToString(layers)}-{time.time()}")
        
        # build model
        model = tf.keras.models.Sequential()
        # add layers
        for nodes in layers:
            model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu))
        # output layer
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        
        # compile and train model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X, y, epochs=epochs, verbose=0, batch_size=512, callbacks=[tensorboard])
        
        # evaluate model
        start_time = time.time()
        result = model.evaluate(X_t, y_t, verbose=0, batch_size=512)
        time_taken = (time.time() - start_time)/len(y_t)
        return result[0], result[1], time_taken
    
    def LSTMModel(X, X_t, y, y_t, epochs, model_name):
        # create callback
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"{ModelContainer.logs_path}{model_name}-{time.time()}")
        
        # build model
        model = tf.keras.models.Sequential()
        # lstm
        model.add(tf.keras.layers.LSTM(128, input_shape=(X.shape[1:])))
        # output
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
        
        # compile and train model
        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X, y, epochs=epochs, verbose=0, batch_size=512, callbacks=[tensorboard])
        
        # evaluate model
        start_time = time.time()
        result = model.evaluate(X_t, y_t, verbose=0, batch_size=512)
        time_taken = (time.time() - start_time)/len(y_t)
        return result[0], result[1], time_taken
    
    