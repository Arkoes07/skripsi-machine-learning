import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import combinations_with_replacement

### Constant ###
NAME = f"{int(time.time())}-ANN"

COMBINATION = (512,256,128,64)

### Functions ### 

# Function which returns subset or r length from n 
def combinationSubset(arr, r): 
    # return list of all subsets of length r 
    return list(combinations_with_replacement(arr, r))

def arrToString(arr):
    text = ""
    for el in arr:
        text = text + '_' + str(el) 
    return text

# Get data which is the output from preprocessing.py
df_class_0 = pd.read_pickle("./preprocessed-data/class_0.pkl")
df_class_1 = pd.read_pickle("./preprocessed-data/class_1.pkl")
df_class_2 = pd.read_pickle("./preprocessed-data/class_2.pkl")

# get the values from each dataframe
datasets = []
datasets.append(df_class_0.values)
datasets.append(df_class_1.values)
datasets.append(df_class_2.values)

# get the minimal length
min_length = min(len(datasets[0]), len(datasets[1]), len(datasets[2]))

### Data Preparation ###
# balance data for each class
for idx in range(3):
    datasets[idx] = random.sample(list(datasets[idx]),min_length)
# flatten dataset
dataset = [item for sublist in datasets for item in sublist]
# shuffle dataset
random.shuffle(dataset)
# separate into X and y values
X = []
y = []
for data in dataset:
    y.append(int(data[0]))
    X.append(data[1:])
# convert into numpy array
X = np.array(X)
y = np.array(y)

### Modelling ANN ###
# generate hidden layer variation
variation = []
variation.extend([item] for item in COMBINATION)
for l in range(2, len(COMBINATION) + 1):
    variation.extend(combinationSubset(COMBINATION,l))

print(f"there is {len(variation)} combiinations of hidden layer\n")

start_time = time.time()

# loop through hidden layer combination:
for hidden_layer in variation:
    model_name = NAME + arrToString(hidden_layer)
    print(f"---> processing {model_name}")
    # create tensorboard object
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs\\{model_name}")
    # build model
    model = tf.keras.models.Sequential()
    # add layers
    for nodes in hidden_layer:
        model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu))
    # output layer
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    # compile model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # train model
    model.fit(X, y, batch_size=128, epochs=20, validation_split=0.25, callbacks=[tensorboard], verbose=2)

print(f"\n== PROGRAM FINISHED in {time.time() - start_time} seconds == ")