import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img

import cv2
import itertools
import pathlib
import warnings
import os
import random
import time
import gc
from IPython.display import Markdown, display
from PIL import Image
from random import randint
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import keras
from tensorflow import keras
from keras import Sequential
from keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout,SeparableConv2D, Activation, BatchNormalization, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import cv2
from keras import models

video = cv2.VideoCapture(0)

def printmd(string):   
    display(Markdown(string))

np.random.seed(0) # Add random seed of training for reproducibility

def load_images_from_folder(folder,only_path = False, label = ""):
    if only_path == False:
        images = []
        for filename in os.listdir(folder):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images
    else:
        path = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder,filename)
            if img_path is not None:
                path.append([label,img_path])
        return path

currentPath = os.getcwdb().decode("utf-8") 
trainPath = os.path.join(currentPath, 'MainRecognition\\train')
testPath = os.path.join(currentPath, 'MainRecognition\\test')
finalPath = os.path.join(currentPath, 'MainRecognition\\final')

images = []
dirp = trainPath+'\\'
for f in os.listdir(dirp):
    try:
        if "png" in os.listdir(dirp+f)[0]:
            images += load_images_from_folder(dirp+f,True,label = f)
        else: 
            for d in os.listdir(dirp+f):
                images += load_images_from_folder(dirp+f+"/"+d,True,label = f)
    except:
        pass
            
df = pd.DataFrame(images, columns = ["fruit", "path"])

from sklearn.utils import shuffle
df = shuffle(df, random_state = 0)
df = df.reset_index(drop=True)

fruit_names = sorted(df.fruit.unique())
mapper_fruit_names = dict(zip(fruit_names, [t for t in range(len(fruit_names))]))
df["label"] = df["fruit"].map(mapper_fruit_names)
print(mapper_fruit_names)

def cut_df(df, number_of_parts, part):

    if part < 1:
        print("Error, the part should be at least 1")
    elif part > number_of_parts:
        print("Error, the part cannot be higher than the number_of_parts")
        
    number_imgs_each_part = int(df.shape[0]/number_of_parts)
    idx1 = (part-1) * number_imgs_each_part
    idx2 = part * number_imgs_each_part
    return df.iloc[idx1:idx2]

def load_img(df):

    img_paths = df["path"].values
    img_labels = df["label"].values
    X = []
    y = []
    
    for i,path in enumerate(img_paths):
        img =  plt.imread(path)
        img = cv2.resize(img, (150,150))
        label = img_labels[i]
        X.append(img)
        y.append(label)
    return np.array(X),np.array(y)

def from_categorical(lst):
    
    lst = lst.tolist()
    lst2 = []
    for x in lst:
        lst2.append(x.index(max(x)))
    return lst2

def display_stats(y_test, pred):
    print(f"### Result of the predictions using {len(y_test)} test data ###\n")
    y_test_class = from_categorical(y_test)
    print("Classification Report:\n")
    print(classification_report(y_test_class, pred))
    print("\nConfusion Matrix:\n\n")
    print(confusion_matrix(y_test_class, pred))
    print("\n")
    printmd(f"# Accuracy: {round(accuracy_score(y_test_class, pred),5)}")

model = models.load_model(currentPath+'\\fruit-recognition_2.h5')

# X, y = load_img(cut_df(df, 20, 20))
# print(y)
# pred = model.predict(X)
# print(pred)
# pred = np.argmax(model.predict(X), axis=1)
# print(pred)
# y_test = to_categorical(y)
# print(y_test)

# display_stats(y_test, pred)
# print(mapper_fruit_names)
class_names = ['Apple', 'Banana', 'Green Apple', 'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Tomatoes']
while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((150,150))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        result = model.predict(img_array, verbose=0)
        # pred = np.argmax(model.predict(img_array), axis=1)
        prediction = int(result[0][0])
        score = tf.nn.softmax(result[0])
        # print(score)
        if(100 * np.max(score)) > 20:
            print(class_names[np.argmax(score)] + " ("+str(100 * np.max(score))+"%)")
        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()